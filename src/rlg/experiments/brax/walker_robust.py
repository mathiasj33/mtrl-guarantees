import copy
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite.walker import PlanarWalker
from mujoco_playground._src.mjx_env import render_array


class WalkerTaskParams(NamedTuple):
    mass_scale: jax.Array  # float
    size_scale: jax.Array  # float


class WalkerRobust(PlanarWalker):
    """Walker environment with robust task variations (Mass, Body)."""

    def __init__(self, move_speed: float = 6.0):
        """
        Args:
            move_speed: Target speed.
        """
        cfg = config_dict.create(
            ctrl_dt=0.025,
            sim_dt=0.0025,
            episode_length=1000,
            action_repeat=1,
            vision=False,
            impl="jax",
            nconmax=50_000,
            njmax=100,
        )
        super().__init__(move_speed, cfg)

        # Cache default physics parameters for scaling
        self._default_body_mass = self.mjx_model.body_mass
        self._default_body_inertia = self.mjx_model.body_inertia
        self._default_dof_damping = self.mjx_model.dof_damping
        self._default_geom_size = self.mjx_model.geom_size

        # Identify the 'torso' geom to represent 'head size' scaling
        # In standard Walker, the torso is the main upper body.
        torso_id = self.mj_model.geom("torso").id
        # Find geoms belonging to the torso body
        self._torso_geom_ids = [
            i
            for i in range(self.mj_model.ngeom)
            if self.mj_model.geom_bodyid[i] == torso_id
        ]
        self._torso_geom_ids = jp.array(self._torso_geom_ids)

    def _gen_model(self, task_params: WalkerTaskParams) -> mjx.Model:
        """Generates a new MJX model with scaled physics parameters.

        Args:
            task_params: [mass_scale, size_scale, damping_scale]
        """
        # 1. Scale Mass and Inertia
        new_mass = self._default_body_mass * task_params.mass_scale
        new_inertia = self._default_body_inertia * task_params.mass_scale

        # 3. Scale Head Size (Torso Geoms)
        # We update the geom_size for torso geoms.
        # geom_size is (ngeom, 3), we scale relevant rows.
        # Note: Changing collision geometry size in MJX is supported
        # but requires care if shapes change drastically.
        new_geom_size = self._default_geom_size

        def scale_geom(sizes, idx):
            return sizes.at[idx, 1].set(sizes[idx, 1] * task_params.size_scale)

        # Apply scaling to all torso geoms
        for geom_id in self._torso_geom_ids:
            # Depending on geom type (capsule/sphere), scaling all dims
            # usually works for "size".
            new_geom_size = scale_geom(new_geom_size, geom_id)

        # Return the new model with replaced parameters
        return self.mjx_model.replace(
            body_mass=new_mass,
            body_inertia=new_inertia,
            geom_size=new_geom_size,
        )

    def reset(self, rng: jax.Array, task_params: WalkerTaskParams) -> mjx_env.State:
        """Resets the environment using the specified task model."""
        # Standard reset logic...
        rng, rng1, rng2 = jax.random.split(rng, 3)

        model = self._gen_model(task_params)
        qpos = jp.zeros(model.nq)
        qpos = qpos.at[2].set(jax.random.uniform(rng1, (), minval=-jp.pi, maxval=jp.pi))

        # Use 'm' (task model) for limits to ensure consistency if limits changed
        # (Though usually limits don't change in this task)
        lowers = model.jnt_range[3:, 0]
        uppers = model.jnt_range[3:, 1]

        qpos = qpos.at[3:].set(
            jax.random.uniform(
                rng2,
                (model.nq - 3,),
                minval=lowers,
                maxval=uppers,
            )
        )

        # CRITICAL: make_data with the specific task model 'm'
        data = mjx_env.make_data(
            model,
            qpos=qpos,
            impl=model.impl.value,
            nconmax=self._config.nconmax,  # type: ignore
            njmax=self._config.njmax,  # type: ignore
        )

        # Forward kinematics with task model 'm'
        data = mjx.forward(model, data)

        metrics = {
            "reward/standing": jp.zeros(()),
            "reward/upright": jp.zeros(()),
            "reward/stand": jp.zeros(()),
            "reward/move": jp.zeros(()),
        }
        info = {"rng": rng, "model": model, "task_params": task_params}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        obs = self._augment_obs_with_task(obs, task_params=task_params)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Steps the simulation using the specified task model."""
        data = mjx_env.step(state.info["model"], state.data, action, self.n_substeps)

        # Pass 'm' if get_reward needs model parameters (optional,
        # here reward is mostly kinematic)
        reward = self._get_reward(data, action, state.info, state.metrics)

        obs = self._get_obs(data, state.info)
        obs = self._augment_obs_with_task(obs, task_params=state.info["task_params"])
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _augment_obs_with_task(self, obs, task_params: WalkerTaskParams):
        task_array = jp.array([task_params.mass_scale, task_params.size_scale])
        return jp.concatenate([obs, jp.log(task_array)], axis=-1)

    def render(
        self,
        task_params: WalkerTaskParams,
        trajectory,
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
        scene_option=None,
        modify_scene_fns=None,
    ):
        # 2. Create a temporary CPU model for visualization
        # We deepcopy to avoid permanently modifying the base environment's model
        render_model = copy.deepcopy(self.mj_model)

        # 3. Apply the visual changes to the CPU model
        # Convert JAX array to numpy for MuJoCo C-structs
        params = np.array([task_params.mass_scale, task_params.size_scale])
        mass_scale, size_scale = params

        # Apply Geometry Scaling (Mirroring gen_model logic)
        # We update the geom_size for the torso geoms on the CPU model
        if size_scale != 1.0:
            for geom_id in self._torso_geom_ids:
                # geom_id is a JAX array in gen_model, but int here.
                # Convert if necessary, or ensure _torso_geom_ids is a standard list/array
                gid = int(geom_id)
                render_model.geom_size[gid, 1] *= size_scale

        # (Optional) Visualize Mass: Tint the torso redder if it's heavier
        # This helps you visually confirm the agent is "heavy"
        if mass_scale != 1.0:
            for geom_id in self._torso_geom_ids:
                gid = int(geom_id)
                # Mix in some red based on mass_scale
                # geom_rgba is [r, g, b, a]
                render_model.geom_rgba[gid, 0] = np.clip(0.5 * mass_scale, 0, 1)

        # 4. Pass the modified model to the utility function
        return render_array(
            render_model,  # <--- Use the modified CPU model
            trajectory,
            height,
            width,
            camera,
            scene_option=scene_option,
            modify_scene_fns=modify_scene_fns,
        )
