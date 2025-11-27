"""An implementation of the zone environment introduced by LTL2Action (Vaezipoor et al., 2021).

The environment simulates a point-mass agent moving in a 2D plane. The agent
applies a forward force aligned with its current heading and can control its
angular velocity. The world contains colored zones that the agent can enter.
The agent is equipped with a lidar sensor that detects the distance to the
nearest zone of each color in a set of evenly spaced angular bins.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jaxltl.environments import environment, spaces
from jaxltl.environments.zone_env.plotter import draw_trajectories
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import (
    EmptyNode,
    MultiOrNode,
    Node,
    NotNode,
    VarNode,
)

if TYPE_CHECKING:
    from jaxltl.environments.renderer.renderer import BaseRenderer

_EPS = 1e-8
_MAX_SAMPLING_ITERS = 1000


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    # World
    world_size: float
    spawn_size: float
    # Zones
    zone_radius: float
    zones_per_color: int
    keepout_radius: float
    # Lidar
    num_lidar_bins: int
    exp_gain: float
    # Physics
    dt: float
    drag: float
    max_speed: float
    max_force: float
    max_angular_velocity: float


class EnvState(eqx.Module):
    # Physics
    position: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angle: jax.Array  # shape: ()
    angular_velocity: jax.Array  # float
    acceleration: jax.Array  # shape: (2,)
    # Zones (static for an episode)
    zone_centers: jax.Array  # shape: (N, 2)
    zone_colors: jax.Array  # shape: (N,) int in [0, C)


class ObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    lidar: jax.Array  # shape: (C, num_bins)


class ResetOptions(NamedTuple):
    pass


class ZoneEnv(environment.Environment[EnvState, EnvParams, ObsFeatures, ResetOptions]):
    default_params = EnvParams(
        max_steps_in_episode=1000,
        world_size=6.6,
        spawn_size=5.0,
        zone_radius=0.4,
        zones_per_color=2,
        keepout_radius=0.55,
        num_lidar_bins=16,
        exp_gain=0.5,
        dt=0.05,
        drag=0.08,
        max_speed=3.0,
        max_force=2.0,
        max_angular_velocity=3.0,
    )
    propositions = ("red", "green", "purple", "yellow")
    max_nodes = 7
    max_edges = 4

    def __init__(self, **kwargs):
        params = dataclasses.asdict(self.default_params) | kwargs
        super().__init__(
            default_params=EnvParams(**params),
            propositions=self.propositions,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )

    @override
    def _observation_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=jnp.inf,
            shape=(5 + len(self.propositions) * int(params.num_lidar_bins),),
            dtype=jnp.float32,
        )

    @override
    def _action_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Box(
            low=jnp.array(
                [-params.max_force, -params.max_angular_velocity], dtype=jnp.float32
            ),
            high=jnp.array(
                [params.max_force, params.max_angular_velocity], dtype=jnp.float32
            ),
            shape=(2,),
            dtype=jnp.float32,
        )

    @override
    def _reset(
        self,
        key_angle: jax.Array,
        state: EnvState | None,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        key_zones, key_pos, key_angle = jax.random.split(key_angle, 3)
        centers, colors = self._sample_zones(key_zones, params)
        agent_pos = self._sample_agent_position(key_pos, params, centers)

        velocity = jnp.zeros(2, dtype=jnp.float32)
        acceleration = jnp.zeros(2, dtype=jnp.float32)
        angle = jax.random.uniform(key_angle, shape=(), minval=-jnp.pi, maxval=jnp.pi)
        angular_velocity = jnp.zeros((), dtype=jnp.float32)

        return EnvState(
            position=agent_pos,
            velocity=velocity,
            angle=angle,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
            zone_centers=centers,
            zone_colors=colors,
        )

    @override
    def _cheap_reset(
        self,
        key: jax.Array,
        state: EnvState,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        raise NotImplementedError("Cheap reset is not implemented for ZoneEnv.")

    def _sample_zones(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, jax.Array]:
        """Sample non-overlapping zone centers and assign colors.

        Returns (centers:(Z,2), colors:(Z,))
        """
        num_colors = len(self.propositions)
        total_zones = num_colors * params.zones_per_color
        minval = -params.spawn_size / 2 + params.keepout_radius
        maxval = params.spawn_size / 2 - params.keepout_radius
        keepout = params.keepout_radius

        centers0 = jnp.zeros((total_zones, 2), dtype=jnp.float32)
        colors = jnp.repeat(
            jnp.arange(num_colors, dtype=jnp.int32), params.zones_per_color
        )

        def cond_fun(carry):
            key, centers, count, it = carry
            return jnp.logical_and(count < total_zones, it < _MAX_SAMPLING_ITERS)

        def body_fun(carry):
            key, centers, count, it = carry
            key, sub = jax.random.split(key)
            proposal = jax.random.uniform(sub, (2,), minval=minval, maxval=maxval)
            idxs = jnp.arange(total_zones)
            mask = idxs < count
            dists = jnp.linalg.norm(centers - proposal, axis=1)
            cond_ok = dists >= 2.0 * keepout
            all_ok = jnp.all(jnp.logical_or(~mask, cond_ok))
            centers = lax.cond(
                all_ok,
                lambda c: c.at[count].set(proposal),
                lambda c: c,
                centers,
            )
            count = count + jnp.where(all_ok, 1, 0)
            return key, centers, count, it + 1

        key, centers, count, it = lax.while_loop(
            cond_fun, body_fun, (key, centers0, jnp.int32(0), jnp.int32(0))
        )
        fallback_centers = jnp.array(  # random but fixed fallback
            [
                [-1.60, -0.54],
                [0.82, 0.10],
                [-0.18, -1.12],
                [-1.91, 1.68],
                [1.58, 1.23],
                [1.82, -1.71],
                [-0.70, 0.58],
                [-0.49, 1.83],
            ]
        )
        centers = jnp.where(count < total_zones, fallback_centers, centers)
        return centers, colors

    def _sample_agent_position(
        self, key: jax.Array, params: EnvParams, centers: jax.Array
    ) -> jax.Array:
        minval = -params.spawn_size / 2 + params.keepout_radius
        maxval = params.spawn_size / 2 - params.keepout_radius

        def agent_cond(carry):
            key, pos, it = carry
            dists = jnp.linalg.norm(centers - pos, axis=1)
            return jnp.logical_and(
                jnp.any(dists < params.keepout_radius * 2), it < _MAX_SAMPLING_ITERS
            )

        def agent_body(carry):
            key, _pos, it = carry
            key, sub = jax.random.split(key)
            pos = jax.random.uniform(sub, (2,), minval=minval, maxval=maxval)
            return key, pos, it + 1

        key_init, key = jax.random.split(key)
        init_pos = jax.random.uniform(key_init, (2,), minval=minval, maxval=maxval)
        key, pos, _ = lax.while_loop(
            agent_cond, agent_body, (key, init_pos, jnp.int32(0))
        )
        return pos

    @override
    def _step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        force = jnp.clip(
            action[0] * params.max_force, -params.max_force, params.max_force
        )
        target_angular_velocity = jnp.clip(
            action[1] * params.max_angular_velocity,
            -params.max_angular_velocity,
            params.max_angular_velocity,
        )
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])
        acceleration = heading * force

        velocity = state.velocity + acceleration * params.dt
        velocity *= 1.0 - params.drag

        speed = jnp.linalg.norm(velocity)
        scaling_factor = jnp.clip(params.max_speed / speed, 0.0, 1.0)
        velocity: jax.Array = jnp.where(
            speed > params.max_speed, velocity * scaling_factor, velocity
        )

        position = state.position + velocity * params.dt

        angle = self._wrap_angle(state.angle + target_angular_velocity * params.dt)
        angular_velocity = target_angular_velocity

        reward = jnp.zeros((), dtype=jnp.float32)
        half_size = params.world_size / 2.0
        terminated = jnp.any(jnp.abs(position) > half_size)

        next_state = EnvState(
            position=position,
            velocity=velocity,
            angle=angle,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
            zone_centers=state.zone_centers,
            zone_colors=state.zone_colors,
        )
        return next_state, reward, terminated, {}

    @staticmethod
    def _wrap_angle(angle: jax.Array) -> jax.Array:
        """Wrap angles to the (-pi, pi] interval."""
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @override
    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        """Compute the observation for a given state."""
        lidar = self._compute_lidar(state, params)
        return ObsFeatures(
            acceleration=state.acceleration,
            velocity=state.velocity,
            angular_velocity=state.angular_velocity.reshape(1),
            lidar=lidar,
        )

    def _compute_lidar(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute per-color lidar distances with evenly spaced bins around the agent.

        Returns an array of shape (C, num_bins) with distances in world units.
        """
        pos = state.position  # (2,)
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])  # (2,)

        centers = state.zone_centers  # (N,2)
        colors = state.zone_colors  # (N,)

        def zone_sensor_binned(zone_pos: jax.Array) -> jax.Array:
            """Compute the sensor of a single zone.

            Returns: (num_bins,)"""

            direction = zone_pos - pos  # (2,)
            dist: float = jnp.linalg.norm(direction)  # ()
            sensor = jnp.exp(-params.exp_gain * dist)
            direction = direction / (dist + _EPS)  # (2,)
            dotp = jnp.dot(heading, direction)
            cross = jnp.cross(heading, direction)
            angle = jnp.arctan2(cross, dotp) % (2.0 * jnp.pi)
            bin_idx = jnp.floor(angle / bin_size).astype(jnp.int32)
            bin_angle = bin_size * bin_idx
            bins = jnp.zeros((params.num_lidar_bins,), dtype=jnp.float32)
            alias = (angle - bin_angle) / bin_size
            bins = bins.at[bin_idx].set(sensor)
            bins = bins.at[bin_idx + 1].set(sensor * alias)
            bins = bins.at[bin_idx - 1].set(sensor * (1.0 - alias))
            return bins

        sensors = jax.vmap(zone_sensor_binned, in_axes=0)(centers)  # (N, num_bins)

        def compute_color_lidar(color_id: jax.Array) -> jax.Array:
            """Compute lidar for a single color."""
            mask_color = colors == color_id  # (num_zones,)
            sensors_color = jnp.where(
                mask_color[:, None], sensors, 0.0
            )  # (N, num_bins)
            sensors_color = jnp.max(sensors_color, axis=0)  # (num_bins,)
            return sensors_color

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        lidar = jax.vmap(compute_color_lidar)(color_ids)  # (C, num_bins)
        return lidar

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute which zones the agent is currently inside.

        Returns an int32 array of shape (C,) containing the color ids of the zones
        the agent is inside, with -1 as padding.
        """
        pos = state.position  # (2,)
        centers = state.zone_centers  # (N,2)
        colors = state.zone_colors  # (N,)

        dists = jnp.linalg.norm(centers - pos, axis=1)  # (N,)
        inside = dists < params.zone_radius  # (N,)

        def compute_color_prop(color_id: jax.Array) -> jax.Array:
            mask_color = colors == color_id  # (N,)
            inside_color = jnp.logical_and(mask_color, inside)  # (N,)
            return jax.lax.cond(jnp.any(inside_color), lambda: color_id, lambda: -1)

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        propositions = jax.vmap(compute_color_prop)(color_ids)  # (C,)
        return jnp.sort(propositions, descending=True)

    @property
    @override
    def assignments(self) -> list[Assignment]:
        """Returns all possible assignments in the environment."""
        assignments = [Assignment(frozenset({color})) for color in self.propositions]
        assignments.append(Assignment(frozenset()))  # empty assignment
        return assignments

    @override
    def assignments_to_graph(self, assignments: frozenset[Assignment]) -> Node | None:
        """Converts a set of assignments to a simplified boolean formula graph for ZoneEnv."""
        if not assignments:
            return None

        if assignments == {Assignment(frozenset())}:
            return EmptyNode()

        # Heuristic 1: Simple VarNode or MultiOrNode
        props_in_set = []
        is_simple_disjunction = True
        for assign in assignments:
            if len(assign.true_propositions) == 1:
                props_in_set.append(list(assign.true_propositions)[0])
            else:
                is_simple_disjunction = False
                break

        if is_simple_disjunction:
            if len(props_in_set) == 1:
                return VarNode(props_in_set[0])
            if len(props_in_set) > 1:
                return MultiOrNode([VarNode(p) for p in sorted(props_in_set)])

        # Heuristic 2: Simple NotNode
        all_assignments = frozenset(self.assignments)
        if len(assignments) == len(all_assignments) - 1:
            missing_assignment = next(iter(all_assignments - assignments))
            if len(missing_assignment.true_propositions) == 1:
                prop = list(missing_assignment.true_propositions)[0]
                return NotNode(VarNode(prop))

        # Fallback to canonical DNF for complex cases
        return self._assignments_to_dnf(assignments)

    @override
    def get_renderer(
        self, env_params: EnvParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures, ResetOptions]":
        """Returns a renderer for the environment."""
        from .renderer import Renderer

        return Renderer(env_params, **kwargs)

    @override
    def plot_trajectories(
        self,
        trajs: EnvState,
        lengths: jax.Array,
        params: EnvParams,
        **plotting_kwargs,
    ) -> None:
        """Plots trajectories of environment states.

        Args:
            trajs: Batched EnvStates of shape (num_episodes, max_length, ...)
            params: Environment parameters
            plotting_kwargs: Additional keyword arguments for the plotting function
        """
        zone_positions = trajs.zone_centers[:, 0].tolist()
        zone_colors = trajs.zone_colors[:, 0].tolist()
        zone_colors = [[self.propositions[i] for i in cs] for cs in zone_colors]
        paths = [
            trajs.position[i, : lengths[i]].tolist() for i in range(lengths.shape[0])
        ]
        draw_trajectories(zone_positions, zone_colors, paths, **plotting_kwargs)
