"""A simple BridgeWorld environment.

The environment consists of a 2D grid world. The agent must navigate the grid to cross a bridge.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxltl.environments import environment, spaces
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import (
    Node,
)

if TYPE_CHECKING:
    from jaxltl.environments.renderer.renderer import BaseRenderer


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    grid_size: int
    bridge_width: int
    bridge_start_row: int
    bridge_end_row: int
    wind_dist: distrax.Distribution


class EnvState(eqx.Module):
    position: jax.Array  # shape: (2,)
    wind_p: jax.Array  # shape: ()


class ObsFeatures(NamedTuple):
    # shape: (2,) agent position
    features: jax.Array


class ResetOptions(NamedTuple):
    pass


class BridgeWorld(
    environment.Environment[EnvState, EnvParams, ObsFeatures, ResetOptions]
):
    propositions = ()
    max_nodes = 0
    max_edges = 0
    _index_to_action = jnp.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32
    )  # right, down, left, up

    def __init__(self, **kwargs):
        default_params = EnvParams(
            max_steps_in_episode=100,
            grid_size=21,
            bridge_width=7,
            bridge_start_row=5,
            bridge_end_row=15,
            wind_dist=distrax.Uniform(low=0.0, high=0.4),
        )
        params = dataclasses.asdict(default_params) | kwargs

        if params["grid_size"] % 2 == 0:
            raise ValueError("grid_size must be odd.")
        if params["bridge_width"] % 2 == 0:
            raise ValueError("bridge_width must be odd.")
        super().__init__(
            default_params=EnvParams(**params),
            propositions=self.propositions,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )

    @override
    def _observation_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Box(
            low=0,
            high=params.grid_size - 1,
            shape=(2,),
            dtype=jnp.int32,
        )

    @override
    def _action_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Discrete(n=4)

    @override
    def _reset(
        self,
        key: jax.Array,
        state: EnvState | None,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        wind_key, reset_key = jax.random.split(key)
        wind_p = params.wind_dist.sample(seed=wind_key)
        init_position = jnp.array(
            [params.grid_size - 1, params.grid_size // 2], dtype=jnp.int32
        )  # bottom-center
        return EnvState(position=init_position, wind_p=wind_p)

    @override
    def _cheap_reset(
        self,
        key: jax.Array,
        state: EnvState,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        return self._reset(key, state, params, options)

    @override
    def _step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        bridge_center = params.grid_size // 2
        bridge_half_width = params.bridge_width // 2
        bridge_start = bridge_center - bridge_half_width
        bridge_end = bridge_center + bridge_half_width

        on_bridge = (state.position[1] >= bridge_start) & (
            state.position[1] <= bridge_end
        )
        on_bridge = jnp.logical_and(
            on_bridge,
            (state.position[0] >= params.bridge_start_row)
            & (state.position[0] <= params.bridge_end_row),
        )

        move = self._index_to_action[action]

        def move_with_wind(key, move, wind_p):
            return jax.lax.cond(
                jax.random.uniform(key) < wind_p,
                lambda: jnp.array([0, -1]),  # move left
                lambda: move,
            )

        wind_p = jnp.where(
            (action == 1) | (action == 3),
            state.wind_p,
            jnp.min(jnp.array((state.wind_p, 0.05))),
        )
        wind_p = jnp.clip(wind_p, 0.0, 1.0)
        move = jax.lax.cond(
            on_bridge,
            lambda: move_with_wind(key, move, wind_p),
            lambda: move,
        )

        pos = state.position + move
        pos = jnp.clip(pos, 0, params.grid_size - 1)

        in_abyss = (
            ((pos[1] < bridge_start) | (pos[1] > bridge_end))
            & (pos[0] >= params.bridge_start_row)
            & (pos[0] <= params.bridge_end_row)
        )
        goal_reached = pos[0] == 0

        reward = jax.lax.cond(
            goal_reached,
            lambda: 1.0,
            lambda: jax.lax.cond(in_abyss, lambda: -1.0, lambda: 0.0),
        )
        terminated = goal_reached | in_abyss

        next_state = EnvState(position=pos, wind_p=state.wind_p)
        return (
            next_state,
            reward,
            terminated,
            {},
        )

    @override
    def get_renderer(
        self, env_params: EnvParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures, ResetOptions]":
        """Returns a renderer for the environment."""
        from .renderer import BridgeWorldRenderer

        return BridgeWorldRenderer(
            title="BridgeWorld",
            screen_size=600,
            grid_size=env_params.grid_size,
            bridge_width=env_params.bridge_width,
            bridge_start_row=env_params.bridge_start_row,
            bridge_end_row=env_params.bridge_end_row,
        )

    @override
    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        return ObsFeatures(features=state.position)

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        return jnp.array([], dtype=jnp.bool)

    @property
    @override
    def assignments(self) -> list[Assignment]:
        """Returns all possible assignments in the environment."""
        return []

    @override
    def assignments_to_graph(self, assignments: frozenset[Assignment]) -> Node | None:
        raise NotImplementedError()

    @override
    def plot_trajectories(
        self,
        trajs: EnvState,
        lengths: jax.Array,
        params: EnvParams,
        **plotting_kwargs,
    ) -> None:
        raise NotImplementedError()
