"""An implementation of the RGB zone environment.

The environment simulates a point-mass agent moving in a 2D plane. The agent applies a
forward force aligned with its current heading and can control its angular velocity. The
world contains RGB-colored zones that the agent can enter. The agent is equipped with
either a lidar sensor that detects the distance and RGB of the nearest zone of each
color in a set of evenly spaced angular bins, or privileged information of the RGBs
and relative positions of each zone.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jaxltl.environments import environment, spaces

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
    num_idxs: int  # I
    zones_per_idx: int  # N
    keepout_radius: float
    # Exteroception
    exteroception: str  # "lidar", "rgb_lidar" or "privileged"
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
    zone_centers: jax.Array  # shape: (N*I, 2)
    zone_idxs: jax.Array  # shape: (N*I,) in [0, I)
    zone_rgbs: jax.Array  # shape: (N*I, 3) in [0, 1]


class LidarObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    lidar: jax.Array  # shape: (I, num_bins)


class RGBLidarObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    rgb_lidar: jax.Array  # shape: (num_bins, 5) -> (r,g,b,intensity,detected)


class PrivilegedObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    privileged: jax.Array  # shape: (N*I, 6) -> (r,g,b,intensity,sin,cos)


ObsFeatures = LidarObsFeatures | RGBLidarObsFeatures | PrivilegedObsFeatures


class ResetOptions(NamedTuple):
    rgb_limits: jax.Array  # shape: (I, 3, 2) in [0, 1]


class RGBZoneEnv(
    environment.Environment[EnvState, EnvParams, ObsFeatures, ResetOptions]
):
    default_params = EnvParams(
        max_steps_in_episode=1000,
        world_size=6.6,
        spawn_size=5.0,
        zone_radius=0.4,
        num_idxs=4,
        zones_per_idx=2,
        keepout_radius=0.55,
        exteroception="rgb_lidar",
        num_lidar_bins=32,
        exp_gain=0.5,
        dt=0.05,
        drag=0.08,
        max_speed=3.0,
        max_force=2.0,
        max_angular_velocity=3.0,
    )
    propositions = ("0", "1", "2", "3")
    max_nodes = 5
    max_edges = 5

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
        proprioception_dims = 5  # accel(2), vel(2), angular_vel(1)
        if params.exteroception == "lidar":
            extero_dims = len(self.propositions) * params.num_lidar_bins
        elif params.exteroception == "rgb_lidar":
            extero_dims = params.num_lidar_bins * 5
        elif params.exteroception == "privileged":
            total_zones = params.num_idxs * params.zones_per_idx
            extero_dims = total_zones * 6
        else:
            raise ValueError(f"Unknown exteroception type: {params.exteroception}")

        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(proprioception_dims + extero_dims,),
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
        centers, idxs, rgbs = self._sample_zones(key_zones, params, options)
        agent_pos = self._sample_agent_position(key_pos, params, centers, options)

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
            zone_idxs=idxs,
            zone_rgbs=rgbs,
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
        self, key: jax.Array, params: EnvParams, options: ResetOptions | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample non-overlapping zone centers and assign rgbs.

        Returns (centers:(Z,2), rgbs:(Z,3))
        """
        total_zones = params.num_idxs * params.zones_per_idx
        minval = -params.spawn_size / 2 + params.keepout_radius
        maxval = params.spawn_size / 2 - params.keepout_radius
        keepout = params.keepout_radius

        centers0 = jnp.zeros((total_zones, 2), dtype=jnp.float32)
        idxs = jnp.repeat(
            jnp.arange(params.num_idxs, dtype=jnp.int32), params.zones_per_idx
        )

        if options is not None:
            rgb_limits = options.rgb_limits
        else:
            rgb_limits = jnp.zeros((params.num_idxs, 3, 2))
            rgb_limits = rgb_limits.at[:, :, 1].set(1.0)

        key, sub = jax.random.split(key)
        rgbs = jnp.repeat(
            jax.random.uniform(
                sub,
                (params.num_idxs, 3),
                minval=rgb_limits[:, :, 0],
                maxval=rgb_limits[:, :, 1],
            ),
            params.zones_per_idx,
            axis=0,
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
        # fallback_centers = jnp.array(  # random but fixed fallback
        #     [
        #         [-1.60, -0.54],
        #         [0.82, 0.10],
        #         [-0.18, -1.12],
        #         [-1.91, 1.68],
        #         [1.58, 1.23],
        #         [1.82, -1.71],
        #         [-0.70, 0.58],
        #         [-0.49, 1.83],
        #     ]
        # )
        # centers = jnp.where(count < total_zones, fallback_centers, centers)
        return centers, idxs, rgbs

    def _sample_agent_position(
        self,
        key: jax.Array,
        params: EnvParams,
        centers: jax.Array,
        options: ResetOptions | None = None,
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
            zone_idxs=state.zone_idxs,
            zone_rgbs=state.zone_rgbs,
        )
        return next_state, reward, terminated, {}

    @staticmethod
    def _wrap_angle(angle: jax.Array) -> jax.Array:
        """Wrap angles to the (-pi, pi] interval."""
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @override
    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        """Compute the observation for a given state."""
        if params.exteroception == "lidar":
            lidar = self._compute_lidar(state, params)
            return LidarObsFeatures(
                acceleration=state.acceleration,
                velocity=state.velocity,
                angular_velocity=state.angular_velocity.reshape(1),
                lidar=lidar,
            )
        elif params.exteroception == "rgb_lidar":
            rgb_lidar = self._compute_rgb_lidar(state, params)
            return RGBLidarObsFeatures(
                acceleration=state.acceleration,
                velocity=state.velocity,
                angular_velocity=state.angular_velocity.reshape(1),
                rgb_lidar=rgb_lidar,
            )
        elif params.exteroception == "privileged":
            privileged = self._compute_privileged(state, params)
            return PrivilegedObsFeatures(
                acceleration=state.acceleration,
                velocity=state.velocity,
                angular_velocity=state.angular_velocity.reshape(1),
                privileged=privileged,
            )
        else:
            raise ValueError(f"Unknown exteroception type: {params.exteroception}")

    def _compute_lidar(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute per-color lidar distances with evenly spaced bins around the agent.

        Returns an array of shape (I, num_bins) with distances in world units.
        """
        pos = state.position  # (2,)
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])  # (2,)

        centers = state.zone_centers  # (N*I,2)
        idxs = state.zone_idxs  # (N*I,)

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

        sensors = jax.vmap(zone_sensor_binned, in_axes=0)(centers)  # (N*I, num_bins)

        def compute_color_lidar(color_id: jax.Array) -> jax.Array:
            """Compute lidar for a single color."""
            mask_color = idxs == color_id  # (N*I,)
            sensors_color = jnp.where(
                mask_color[:, None], sensors, 0.0
            )  # (N*I, num_bins)
            sensors_color = jnp.max(sensors_color, axis=0)  # (num_bins,)
            return sensors_color

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        lidar = jax.vmap(compute_color_lidar)(color_ids)  # (I, num_bins)
        return lidar

    def _compute_rgb_lidar(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Computes a single lidar with RGB and intensity of the most intense zone.

        Returns: (num_bins, 5) array (r, g, b, intensity, detected)
        """
        pos = state.position
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])

        centers = state.zone_centers
        rgbs = state.zone_rgbs

        direction = centers - pos
        dist = jnp.linalg.norm(direction, axis=1)
        intensity = jnp.exp(-params.exp_gain * dist)

        direction_norm = direction / (dist[:, None] + _EPS)
        dotp = jnp.dot(direction_norm, heading)
        cross = jnp.cross(direction_norm, heading)
        angle = jnp.arctan2(cross, dotp) % (2.0 * jnp.pi)
        bin_idx = jnp.floor(angle / bin_size).astype(jnp.int32)

        # For each bin, find the zone with max intensity
        def max_intensity_in_bin(i):
            mask = bin_idx == i
            intensities_in_bin = jnp.where(mask, intensity, -1.0)
            max_idx = jnp.argmax(intensities_in_bin)
            max_intensity = intensity[max_idx]
            rgb = rgbs[max_idx]
            detected = jnp.any(mask)
            # If not detected, return zeros, otherwise return the values
            return jnp.where(
                detected,
                jnp.concatenate([rgb, jnp.array([max_intensity, 1.0])]),
                jnp.zeros(5),
            )

        bins = jnp.arange(params.num_lidar_bins)
        return jax.vmap(max_intensity_in_bin)(bins)

    def _compute_privileged(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Computes privileged information for each zone.

        Returns: (N*I, 6) array (r, g, b, intensity, sin(angle), cos(angle))
        """
        pos = state.position
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])

        centers = state.zone_centers
        rgbs = state.zone_rgbs

        direction = centers - pos
        dist = jnp.linalg.norm(direction, axis=1)
        intensity = jnp.exp(-params.exp_gain * dist)

        direction_norm = direction / (dist[:, None] + _EPS)
        dotp = jnp.dot(direction_norm, heading)
        cross = jnp.cross(direction_norm, heading)
        angle = jnp.arctan2(cross, dotp)

        sin_angle = jnp.sin(angle)
        cos_angle = jnp.cos(angle)

        return jnp.concatenate(
            [
                rgbs,
                intensity[:, None],
                sin_angle[:, None],
                cos_angle[:, None],
            ],
            axis=1,
        )

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute which zones the agent is currently inside.

        Returns an int32 array of shape (I,) containing the color ids of the zones
        the agent is inside, with -1 as padding.
        """
        pos = state.position  # (2,)
        centers = state.zone_centers  # (N,2)
        idxs = state.zone_idxs  # (N,)

        dists = jnp.linalg.norm(centers - pos, axis=1)  # (N,)
        inside = dists < params.zone_radius  # (N,)

        def compute_color_prop(color_id: jax.Array) -> jax.Array:
            mask_color = idxs == color_id  # (N,)
            inside_color = jnp.logical_and(mask_color, inside)  # (N,)
            return jax.lax.cond(jnp.any(inside_color), lambda: color_id, lambda: -1)

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        propositions = jax.vmap(compute_color_prop)(color_ids)  # (I,)
        return jnp.sort(propositions, descending=True)

    @property
    @override
    def assignments_array(self) -> jax.Array:
        """Returns the possible assignments in the environment.

        Returns: array of shape (num_assignments, I) int32
        """
        color_assignments = jnp.arange(len(self.propositions)).reshape(-1, 1)
        empty_assignment = jnp.array([[-1]], dtype=jnp.int32)
        assignments = jnp.vstack([color_assignments, empty_assignment])
        padding = -jnp.ones(
            (assignments.shape[0], len(self.propositions) - 1),
            dtype=jnp.int32,
        )
        return jnp.hstack([assignments, padding])

    def get_renderer(
        self, env_params: EnvParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures, ResetOptions]":
        """Returns a renderer for the environment."""
        from .renderer import Renderer

        return Renderer(env_params, **kwargs)
