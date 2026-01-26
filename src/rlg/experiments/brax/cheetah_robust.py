import copy
from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite.cheetah import Run
from mujoco_playground._src.mjx_env import render_array


class CheetahTaskParams(NamedTuple):
    mass_scale: jax.Array  # float
    torso_length_scale: jax.Array  # float


class CheetahRobust(Run):
    """Cheetah environment with robust task variations (Mass, Torso Length)."""

    def __init__(self):
        """Initialize the robust Cheetah environment."""
        cfg = config_dict.create(
            ctrl_dt=0.01,
            sim_dt=0.01,
            episode_length=1000,
            action_repeat=1,
            vision=False,
            impl="jax",
            nconmax=100_000,
            njmax=100,
        )
        super().__init__(config=cfg)

        # Cache default physics parameters for scaling
        self._default_body_mass = self.mjx_model.body_mass
        self._default_body_inertia = self.mjx_model.body_inertia
        self._default_geom_size = self.mjx_model.geom_size
        self._default_geom_pos = self.mjx_model.geom_pos
        self._default_body_pos = self.mjx_model.body_pos

        # Identify the 'torso' geom for length scaling
        # The torso geom is a capsule defined with fromto="-.5 0 0 .5 0 0"
        self._torso_geom_id = self.mj_model.geom("torso").id
        self._head_geom_id = self.mj_model.geom("head").id

        # Identify leg bodies that need to move when torso length changes
        # bthigh is at pos="-.5 0 0" (back of torso)
        # fthigh is at pos=".5 0 0" (front of torso)
        self._bthigh_body_id = self.mj_model.body("bthigh").id
        self._fthigh_body_id = self.mj_model.body("fthigh").id

        # Get torso body id for inertia updates
        self._torso_body_id = self.mj_model.body("torso").id

        # Cache torso capsule properties for inertia computation
        # For a capsule: size[0] = radius, size[1] = half-length
        self._torso_radius = float(self._default_geom_size[self._torso_geom_id, 0])
        self._torso_half_length = float(self._default_geom_size[self._torso_geom_id, 1])

    def _compute_capsule_inertia(
        self, mass: jax.Array, radius: float, half_length: jax.Array
    ) -> jax.Array:
        """Compute the diagonal inertia tensor for a capsule aligned along x-axis.

        For a capsule (cylinder + two hemispherical caps):
        - Total length = 2 * half_length (cylinder part) + 2 * radius (caps)
        - We approximate as a cylinder for simplicity, which is reasonable
          when the cylinder part dominates.

        For a cylinder aligned along x-axis:
        - Ixx = 0.5 * m * r^2
        - Iyy = Izz = (1/12) * m * (3*r^2 + L^2) where L = 2*half_length

        Args:
            mass: Mass of the capsule
            radius: Capsule radius
            half_length: Half-length of the cylindrical part

        Returns:
            Inertia tensor as [Ixx, Iyy, Izz]
        """
        r2 = radius**2
        L = 2.0 * half_length
        L2 = L**2

        Ixx = 0.5 * mass * r2
        Iyy = (1.0 / 12.0) * mass * (3.0 * r2 + L2)
        Izz = Iyy  # Symmetric about x-axis

        return jp.array([Ixx, Iyy, Izz])

    def _gen_model(self, task_params: CheetahTaskParams) -> mjx.Model:
        """Generates a new MJX model with scaled physics parameters.

        Args:
            task_params: CheetahTaskParams containing mass_scale and torso_length_scale
        """
        # 1. Scale Mass uniformly for all bodies
        new_mass = self._default_body_mass * task_params.mass_scale

        # 2. Scale Inertia - most bodies just scale with mass,
        #    but torso needs special handling due to length change
        new_inertia = self._default_body_inertia * task_params.mass_scale

        # 3. Scale Torso Length
        # For a capsule defined with fromto, MuJoCo stores:
        #   - size[0] = radius
        #   - size[1] = half-length (computed from fromto endpoints)
        # The original torso has fromto="-.5 0 0 .5 0 0", so half-length = 0.5
        new_geom_size = self._default_geom_size
        new_torso_half_length = self._torso_half_length * task_params.torso_length_scale
        new_geom_size = new_geom_size.at[self._torso_geom_id, 1].set(
            new_torso_half_length
        )

        # 4. Update torso body inertia to reflect the new capsule dimensions
        # The torso inertia should account for both mass_scale and length change
        torso_mass = new_mass[self._torso_body_id]
        new_torso_inertia = self._compute_capsule_inertia(
            torso_mass, self._torso_radius, new_torso_half_length
        )
        new_inertia = new_inertia.at[self._torso_body_id].set(new_torso_inertia)

        # 5. Scale the head geom position along x-axis
        # Original head pos is (0.6, 0, 0.1) in the torso body frame
        new_geom_pos = self._default_geom_pos
        head_pos = self._default_geom_pos[self._head_geom_id]
        new_head_x = head_pos[0] * task_params.torso_length_scale
        new_geom_pos = new_geom_pos.at[self._head_geom_id, 0].set(new_head_x)

        # 6. Scale the leg body positions to match the new torso endpoints
        # bthigh is at pos="-.5 0 0" (back end of torso)
        # fthigh is at pos=".5 0 0" (front end of torso)
        new_body_pos = self._default_body_pos
        bthigh_pos = self._default_body_pos[self._bthigh_body_id]
        fthigh_pos = self._default_body_pos[self._fthigh_body_id]
        new_body_pos = new_body_pos.at[self._bthigh_body_id, 0].set(
            bthigh_pos[0] * task_params.torso_length_scale
        )
        new_body_pos = new_body_pos.at[self._fthigh_body_id, 0].set(
            fthigh_pos[0] * task_params.torso_length_scale
        )

        # Return the new model with replaced parameters
        return self.mjx_model.replace(
            body_mass=new_mass,
            body_inertia=new_inertia,
            geom_size=new_geom_size,
            geom_pos=new_geom_pos,
            body_pos=new_body_pos,
        )

    def reset(self, rng: jax.Array, task_params: CheetahTaskParams) -> mjx_env.State:
        """Resets the environment using the specified task model."""
        rng, rng1 = jax.random.split(rng, 2)

        model = self._gen_model(task_params)
        qpos = jp.zeros(model.nq)
        qpos = qpos.at[3:].set(
            jax.random.uniform(
                rng1,
                (model.nq - 3,),
                minval=self._lowers,
                maxval=self._uppers,
            )
        )

        # Create data with the task-specific model
        data = mjx_env.make_data(
            model,
            qpos=qpos,
            impl=model.impl.value,
            nconmax=self._config.nconmax,  # type: ignore
            njmax=self._config.njmax,  # type: ignore
        )

        # Forward kinematics with task model
        data = mjx.forward(model, data)

        # Stabilize
        # data = mjx_env.step(model, data, jp.zeros(model.nu), 200)
        data = data.replace(time=0.0)

        metrics = {}
        info = {"rng": rng, "model": model, "task_params": task_params}

        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        obs = self._augment_obs_with_task(obs, task_params=task_params)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Steps the simulation using the specified task model."""
        data = mjx_env.step(state.info["model"], state.data, action, self.n_substeps)

        reward = self._get_reward(data, action, state.info, state.metrics)

        obs = self._get_obs(data, state.info)
        obs = self._augment_obs_with_task(obs, task_params=state.info["task_params"])
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _augment_obs_with_task(self, obs, task_params: CheetahTaskParams):
        """Augment observation with log-scaled task parameters."""
        task_array = jp.array([task_params.mass_scale, task_params.torso_length_scale])
        return jp.concatenate([obs, jp.log(task_array)], axis=-1)

    def render(
        self,
        task_params: CheetahTaskParams,
        trajectory,
        height: int = 240,
        width: int = 320,
        camera: str | None = None,
        scene_option=None,
        modify_scene_fns=None,
    ):
        """Render a trajectory with the given task parameters.

        Args:
            task_params: Task parameters for visualization
            trajectory: List of states to render
            height: Frame height
            width: Frame width
            camera: Camera name
            scene_option: MuJoCo scene options
            modify_scene_fns: Functions to modify the scene

        Returns:
            Array of rendered frames
        """
        # Create a temporary CPU model for visualization
        render_model = copy.deepcopy(self.mj_model)

        # Convert JAX array to numpy for MuJoCo C-structs
        mass_scale = float(task_params.mass_scale)
        torso_length_scale = float(task_params.torso_length_scale)

        # Apply torso length scaling
        if torso_length_scale != 1.0:
            torso_id = int(self._torso_geom_id)
            head_id = int(self._head_geom_id)
            bthigh_id = int(self._bthigh_body_id)
            fthigh_id = int(self._fthigh_body_id)

            # Scale torso geom half-length (size[1] for capsule)
            render_model.geom_size[torso_id, 1] *= torso_length_scale

            # Scale head position
            render_model.geom_pos[head_id, 0] *= torso_length_scale

            # Scale leg body positions
            render_model.body_pos[bthigh_id, 0] *= torso_length_scale
            render_model.body_pos[fthigh_id, 0] *= torso_length_scale

        # Visualize mass by tinting the torso
        if mass_scale != 1.0:
            torso_id = int(self._torso_geom_id)
            # Mix in some red based on mass_scale
            render_model.geom_rgba[torso_id, 0] = np.clip(0.5 * mass_scale, 0, 1)

        # Render using the modified model
        return render_array(
            render_model,
            trajectory,
            height,
            width,
            camera,
            scene_option=scene_option,
            modify_scene_fns=modify_scene_fns,
        )
