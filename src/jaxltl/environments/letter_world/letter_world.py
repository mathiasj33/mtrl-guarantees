"""An implementation of the LetterWorld environment introduced by LTL2Action (Vaezipoor et al., 2021).

The environment consists of a 2D grid world with randomly placed letters. The agent must
navigate the grid, as specified by LTL formulas, to visit the letters in a certain order.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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
    letter_freq: int  # how often each letter appears on the grid


class EnvState(eqx.Module):
    position: jax.Array  # shape: (2,)
    letters: jax.Array  # shape: (G, G, L) one-hot encoding of letters on grid


class ObsFeatures(NamedTuple):
    # shape: (G, G, L + 1) letters pos relative to agent and agent pos
    features: jax.Array


class ResetOptions(NamedTuple):
    pass


class LetterWorld(
    environment.Environment[EnvState, EnvParams, ObsFeatures, ResetOptions]
):
    default_params = EnvParams(max_steps_in_episode=75, grid_size=7, letter_freq=2)
    propositions = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l")
    max_nodes = 7
    max_edges = 4
    _index_to_action = jnp.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32
    )  # right, down, left, up

    def __init__(self, **kwargs):
        params = dataclasses.asdict(self.default_params) | kwargs
        if params["grid_size"] % 2 == 0:
            raise ValueError("grid_size must be odd.")
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
            high=1.0,
            shape=(params.grid_size, params.grid_size, len(self.propositions) + 1),
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
        def loop_cond(carry):
            _, letters = carry
            is_valid = self._is_map_valid(letters, params.grid_size)
            return jnp.logical_not(is_valid)

        def loop_body(carry):
            shuffle_key, _ = carry
            shuffle_key, subkey = jax.random.split(shuffle_key)
            letters = self._sample_letters(subkey, params)
            return (shuffle_key, letters)

        shuffle_key, subkey = jax.random.split(key)
        _, letters = jax.lax.while_loop(
            loop_cond,
            loop_body,
            (shuffle_key, self._sample_letters(subkey, params)),
        )
        init_position = jnp.array([0, 0], dtype=jnp.int32)
        return EnvState(position=init_position, letters=letters)

    def _sample_letters(self, shuffle_key: jax.Array, params: EnvParams) -> jax.Array:
        """Samples random letter placements on the grid."""

        num_propositions = len(self.propositions)
        num_items_to_place = num_propositions * params.letter_freq
        total_grid_cells = params.grid_size * params.grid_size

        letter_indices = jnp.repeat(jnp.arange(num_propositions), params.letter_freq)

        # Select random unique positions on the flattened grid (0 to G*G-1)
        flat_positions = jax.random.choice(
            shuffle_key, total_grid_cells, shape=(num_items_to_place,), replace=False
        )

        # Create a flat empty grid: (Total Cells, Num Letters)
        flat_grid = jnp.zeros((total_grid_cells, num_propositions), dtype=jnp.int32)

        # Scatter the values: Set 1s at the chosen positions for specific letter channels
        flat_grid = flat_grid.at[flat_positions, letter_indices].set(1)

        # Reshape back to spatial dimensions: (G, G, L)
        return flat_grid.reshape(params.grid_size, params.grid_size, num_propositions)

    def _is_map_valid(self, letters: jax.Array, grid_size: int) -> jax.Array:
        """
        Checks if all grid cells are reachable from (0, 0).
        Returns a boolean scalar.
        """
        # Shape: (G, G). True if cell has a letter.
        has_letter = jnp.sum(letters, axis=-1) > 0
        can_expand_from = ~has_letter

        # Start with only (0,0) reachable.
        reachable = jnp.zeros((grid_size, grid_size), dtype=jnp.bool)
        reachable = reachable.at[0, 0].set(True)

        # Flood fill
        def expand_step(_, current_reachable):
            sources = jnp.logical_and(current_reachable, can_expand_from)

            # Spread to 4 neighbors with toroidal wrapping
            neighbors = (
                jnp.roll(sources, shift=1, axis=0)  # Down
                | jnp.roll(sources, shift=-1, axis=0)  # Up
                | jnp.roll(sources, shift=1, axis=1)  # Right
                | jnp.roll(sources, shift=-1, axis=1)  # Left
            )

            # A cell is reachable if it was already reachable OR a neighbor spread to it
            return current_reachable | neighbors

        # The max shortest path in a grid is bounded by G*G.
        final_reachable = jax.lax.fori_loop(
            0, grid_size * grid_size, expand_step, reachable
        )

        # Check if ALL cells are reachable
        return jnp.all(final_reachable)

    @override
    def _cheap_reset(
        self,
        key: jax.Array,
        state: EnvState,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        raise NotImplementedError("Cheap reset is not implemented for LetterWorld.")

    @override
    def _step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        pos = state.position + self._index_to_action[action]  # (2,)
        pos %= params.grid_size  # wrap around
        next_state = EnvState(position=pos, letters=state.letters)
        return (
            next_state,
            jnp.zeros((), dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.bool),
            {},
        )

    @override
    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        """Compute the observation for a given state. Observation is centred on the agent's position."""
        center = params.grid_size // 2
        delta = jnp.array([center, center], dtype=jnp.int32) - state.position  # (2,)
        relative_letters = jnp.roll(state.letters, shift=delta, axis=(0, 1))
        agent_channel = jnp.zeros(
            (params.grid_size, params.grid_size, 1), dtype=relative_letters.dtype
        )
        agent_channel = agent_channel.at[center, center, 0].set(1)
        obs = jnp.concatenate([relative_letters, agent_channel], axis=-1)
        return ObsFeatures(features=obs)

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute which letter the agent is currently on (if any).

        Returns an int32 array of shape (num_letters,).
        """
        pos_letters = state.letters[state.position[0], state.position[1], :]  # (L,)
        return jnp.nonzero(pos_letters, size=len(self.propositions), fill_value=-1)[0]

    @property
    @override
    def assignments(self) -> list[Assignment]:
        """Returns all possible assignments in the environment."""
        assignments = [Assignment(frozenset({letter})) for letter in self.propositions]
        assignments.append(Assignment(frozenset()))  # empty assignment
        return assignments

    @override
    def assignments_to_graph(self, assignments: frozenset[Assignment]) -> Node | None:
        raise NotImplementedError()

    @override
    def get_renderer(
        self, env_params: EnvParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures, ResetOptions]":
        """Returns a renderer for the environment."""
        from .renderer import LetterWorldRenderer

        return LetterWorldRenderer(
            title="LetterWorld",
            screen_size=600,
            grid_size=env_params.grid_size,
        )

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
        from .plotter import draw_trajectories

        letters_grids = np.array(trajs.letters[:, 0])
        paths = [
            trajs.position[i, : lengths[i]].tolist() for i in range(lengths.shape[0])
        ]
        draw_trajectories(
            letters_grids,
            paths,
            self.propositions,
            params.grid_size,
            **plotting_kwargs,
        )
