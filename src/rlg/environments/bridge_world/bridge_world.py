"""A simple BridgeWorld environment.

The environment consists of a 2D grid world (10x8). The agent starts in the lower left
quadrant and must navigate to the goal at the top. There are two bridges:
- Left bridge: shorter path, directly ahead
- Right bridge: longer path, requires going right first then left after crossing

Both bridges are 4 squares long and have different wind distributions.
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
    grid_width: int  # number of columns
    grid_height: int  # number of rows
    bridge_length: int  # length of bridges (rows)
    bridge_width: int  # width of bridges (columns)
    left_bridge_col: int  # center column of left bridge
    right_bridge_col: int  # center column of right bridge
    bridge_start_row: int  # row where bridges start
    left_wind_dist: distrax.Distribution  # wind distribution for left bridge
    right_wind_dist: distrax.Distribution  # wind distribution for right bridge


class EnvState(eqx.Module):
    position: jax.Array  # shape: (2,) - (row, col)
    left_wind_p: jax.Array  # shape: () - wind probability for left bridge
    right_wind_p: jax.Array  # shape: () - wind probability for right bridge


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
        # Default: 10x8 grid with two bridges (3 cells wide each)
        # Left bridge centered at column 2, right bridge centered at column 7
        # Bridges span rows 2-5 (4 squares long)
        default_params = EnvParams(
            max_steps_in_episode=50,
            grid_width=10,
            grid_height=8,
            bridge_length=4,
            bridge_width=3,
            left_bridge_col=2,
            right_bridge_col=7,
            bridge_start_row=2,
            left_wind_dist=distrax.Uniform(low=0.2, high=0.3),  # moderate wind
            right_wind_dist=distrax.Uniform(low=0.0, high=0.15),  # lighter wind
        )
        params = dataclasses.asdict(default_params) | kwargs

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
            high=jnp.array([params.grid_height - 1, params.grid_width - 1]),
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
        left_wind_key, right_wind_key = jax.random.split(key)
        left_wind_p = params.left_wind_dist.sample(seed=left_wind_key)
        right_wind_p = params.right_wind_dist.sample(seed=right_wind_key)
        # Start in lower left quadrant (bottom-left area)
        init_position = jnp.array(
            [params.grid_height - 1, 1], dtype=jnp.int32
        )  # bottom row, second column
        return EnvState(
            position=init_position,
            left_wind_p=left_wind_p,
            right_wind_p=right_wind_p,
        )

    @override
    def _cheap_reset(
        self,
        key: jax.Array,
        state: EnvState,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        return self._reset(key, state, params, options)

    def _is_on_bridge(
        self, position: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Check if position is on a bridge.

        Returns:
            (on_any_bridge, on_left_bridge, on_right_bridge)
        """
        row, col = position[0], position[1]
        bridge_end_row = params.bridge_start_row + params.bridge_length - 1
        half_width = params.bridge_width // 2

        in_bridge_rows = (row >= params.bridge_start_row) & (row <= bridge_end_row)

        # Check if within left bridge columns (center +/- half_width)
        left_start = params.left_bridge_col - half_width
        left_end = params.left_bridge_col + half_width
        on_left_bridge = in_bridge_rows & (col >= left_start) & (col <= left_end)

        # Check if within right bridge columns
        right_start = params.right_bridge_col - half_width
        right_end = params.right_bridge_col + half_width
        on_right_bridge = in_bridge_rows & (col >= right_start) & (col <= right_end)

        on_any_bridge = on_left_bridge | on_right_bridge

        return on_any_bridge, on_left_bridge, on_right_bridge

    def _is_in_abyss(self, position: jax.Array, params: EnvParams) -> jax.Array:
        """Check if position is in the abyss (between bridges, not on a bridge)."""
        row, col = position[0], position[1]
        bridge_end_row = params.bridge_start_row + params.bridge_length - 1
        half_width = params.bridge_width // 2

        in_bridge_rows = (row >= params.bridge_start_row) & (row <= bridge_end_row)

        # Check if on left bridge
        left_start = params.left_bridge_col - half_width
        left_end = params.left_bridge_col + half_width
        on_left_bridge = (col >= left_start) & (col <= left_end)

        # Check if on right bridge
        right_start = params.right_bridge_col - half_width
        right_end = params.right_bridge_col + half_width
        on_right_bridge = (col >= right_start) & (col <= right_end)

        # In abyss if in bridge rows but not on either bridge
        return in_bridge_rows & ~on_left_bridge & ~on_right_bridge

    @override
    def _step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        on_bridge, on_left, on_right = self._is_on_bridge(state.position, params)

        move = self._index_to_action[action]

        def apply_wind(key, move, wind_p):
            """Apply wind effect - pushes agent left with probability wind_p."""
            return jax.lax.cond(
                jax.random.uniform(key) < wind_p,
                lambda: jnp.array([0, -1], dtype=jnp.int32),  # blown left
                lambda: move,
            )

        # Determine wind probability based on which bridge
        wind_p = jnp.where(on_left, state.left_wind_p, state.right_wind_p)

        # Only apply wind when moving up or down on bridge
        is_vertical_move = (action == 1) | (action == 3)
        wind_p = jnp.where(is_vertical_move, wind_p, 0.0)

        # Apply wind effect only when on bridge
        move = jax.lax.cond(
            on_bridge,
            lambda: apply_wind(key, move, wind_p),
            lambda: move,
        )

        pos = state.position + move
        pos = jnp.clip(
            pos,
            jnp.array([0, 0]),
            jnp.array([params.grid_height - 1, params.grid_width - 1]),
        )

        in_abyss = self._is_in_abyss(pos, params)

        # Goal is only in front of left bridge (top row, within left bridge columns)
        half_width = params.bridge_width // 2
        left_start = params.left_bridge_col - half_width
        left_end = params.left_bridge_col + half_width
        goal_reached = (pos[0] == 0) & (pos[1] >= left_start) & (pos[1] <= left_end)

        reward = jax.lax.cond(
            goal_reached,
            lambda: 1.0,
            lambda: jax.lax.cond(in_abyss, lambda: -1.0, lambda: 0.0),
        )
        terminated = goal_reached | in_abyss

        next_state = EnvState(
            position=pos,
            left_wind_p=state.left_wind_p,
            right_wind_p=state.right_wind_p,
        )
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
            grid_width=env_params.grid_width,
            grid_height=env_params.grid_height,
            bridge_length=env_params.bridge_length,
            bridge_width=env_params.bridge_width,
            left_bridge_col=env_params.left_bridge_col,
            right_bridge_col=env_params.right_bridge_col,
            bridge_start_row=env_params.bridge_start_row,
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
