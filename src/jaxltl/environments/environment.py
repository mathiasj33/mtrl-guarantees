"""Abstract base class for all jaxltl environments.

Adapted from gymnax (https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/environment.py).
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.spaces import Space
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import (
    EmptyNode,
    MultiAndNode,
    MultiOrNode,
    Node,
    NotNode,
    VarNode,
)

if TYPE_CHECKING:
    from jaxltl.environments.renderer.renderer import BaseRenderer


@dataclass(frozen=True)
class EnvParams:
    """Base class for environment parameters.

    Note: changing environment parameters will require recompilation of jitted functions.
    """

    max_steps_in_episode: int


class EnvObservation[TObsFeatures: NamedTuple](eqx.Module):
    """Environment observation. Can be extended by wrappers to add additional fields."""

    features: TObsFeatures


class EnvTransition[TEnvState: eqx.Module, TObsFeatures: NamedTuple](NamedTuple):
    """Environment transition."""

    state: TEnvState
    observation: EnvObservation[TObsFeatures]
    reward: jax.Array  # shape: ()
    terminated: jax.Array  # shape: () boolean
    truncated: jax.Array  # shape: () boolean
    terminal_observation: EnvObservation[TObsFeatures]  # used if done
    # shape: (num_propositions,) int32: index in propositions / -1 for padding
    propositions: jax.Array
    info: dict[Any, Any]

    @property
    def done(self) -> jax.Array:
        """Whether the episode is done (terminated or truncated)."""
        return jnp.logical_or(self.terminated, self.truncated)


class Environment[
    TEnvState: eqx.Module,
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](eqx.Module):
    """Abstract base class for environments."""

    default_params: TEnvParams
    # Maps indices in obs.propositions to names
    propositions: tuple[str, ...]
    assignments_array: jax.Array  # shape: (num_assignments, num_propositions) int32

    max_nodes: int  # max. nodes of boolean formula graphs for this environment
    max_edges: int  # max. edges of boolean formula graphs for this environment

    def __init__(
        self,
        default_params: TEnvParams,
        propositions: tuple[str, ...],
        max_nodes: int = 5,
        max_edges: int = 5,
    ):
        self.default_params = default_params
        self.propositions = propositions
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.assignments_array = self._compute_assignments_array()

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=2)
    def reset(
        self,
        key: jax.Array,
        state: TEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        """Performs resetting of environment.

        Dependence on state is needed for some wrappers (e.g. CurriculumWrapper).
        """
        state = self._reset(key, state, params, options)
        return state, self.compute_obs(state, params)

    @abstractmethod
    def _reset(
        self,
        key: jax.Array,
        state: TEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> TEnvState:
        """Environment-specific reset."""
        pass

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=2)
    def cheap_reset(
        self,
        key: jax.Array,
        state: TEnvState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TEnvState, EnvObservation[TObsFeatures]]:
        """Performs a cheap reset of the environment given the current state.
        Since JIT requires resetting on every step, this method can be used to implement
        a faster reset to improve performance. See AutoResetWrapper for further details.
        """

        state = self._cheap_reset(key, state, params, options)
        return state, self.compute_obs(state, params)

    @abstractmethod
    def _cheap_reset(
        self,
        key: jax.Array,
        state: TEnvState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> TEnvState:
        """Environment-specific cheap reset."""
        pass

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=2)
    def step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[TEnvState, TObsFeatures]:
        """Performs step transitions in the environment."""
        next_state, reward, terminated, info = self._step(key, state, action, params)
        obs = self.compute_obs(next_state, params)
        propositions = self.compute_propositions(next_state, params)
        transition = EnvTransition(
            state=next_state,
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=jnp.array(False, dtype=jnp.bool),
            terminal_observation=obs,
            propositions=propositions,
            info=info,
        )
        return jax.lax.stop_gradient(transition)

    @abstractmethod
    def _step(
        self,
        key: jax.Array,
        state: TEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> tuple[TEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Environment-specific step transition.
        Returns: next_state, reward, terminated, info"""
        pass

    def compute_obs(
        self, state: TEnvState, params: TEnvParams
    ) -> EnvObservation[TObsFeatures]:
        """Compute the observation for a given state."""
        return EnvObservation(features=self._compute_obs(state, params))

    @abstractmethod
    def _compute_obs(self, state: TEnvState, params: TEnvParams) -> TObsFeatures:
        """Compute the environment-specific observation for a given state."""
        pass

    @abstractmethod
    def compute_propositions(self, state: TEnvState, params: TEnvParams) -> jax.Array:
        """Computes atomic propositions from environment state.

        Returns: int32 array of shape (num_propositions,) where each entry is the index
        in self.propositions (or -1 for padding)."""
        pass

    def observation_space(self, params: TEnvParams | None = None) -> Space:
        """Observation space of the environment."""
        if params is None:
            params = self.default_params
        return self._observation_space(params)

    @abstractmethod
    def _observation_space(self, params: TEnvParams) -> Space:
        pass

    def action_space(self, params: TEnvParams | None = None) -> Space:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return self._action_space(params)

    @abstractmethod
    def _action_space(self, params: TEnvParams) -> Space:
        pass

    def map_assignment_to_index(self, assignment: jax.Array) -> jax.Array:
        """Maps a proposition assignment to an index in the assignments array.

        Args:
            assignment: jax.Array of shape (num_propositions,) int32

        Returns:
            jax.Array of shape () int32: index in assignments array
        """
        # (num_assignments,)
        assignment = jnp.sort(assignment, descending=True)
        matches = jnp.all(self.assignments_array == assignment, axis=1)
        return jnp.argmax(matches)  # () int32

    def _assignments_to_dnf(self, assignments: frozenset[Assignment]) -> Node | None:
        """Converts a set of assignments to a boolean formula (graph) in DNF."""
        if not assignments:
            return None

        assignment_conjuncts = []
        for assignment in assignments:
            literals = []
            for prop in self.propositions:
                if prop in assignment:
                    literals.append(VarNode(prop))
                else:
                    literals.append(NotNode(VarNode(prop)))

            if not literals:
                assignment_conjuncts.append(EmptyNode())
                continue
            if len(literals) == 1:
                assignment_conjuncts.append(literals[0])
            else:
                assignment_conjuncts.append(MultiAndNode(literals))

        if not assignment_conjuncts:
            return None
        if len(assignment_conjuncts) == 1:
            return assignment_conjuncts[0]
        return MultiOrNode(assignment_conjuncts)

    def assignments_to_graph(self, assignments: frozenset[Assignment]) -> Node | None:
        """Converts a set of assignments to a boolean formula graph.

        This base implementation creates a canonical DNF representation.
        Environments can override this for a more simplified representation.
        """
        return self._assignments_to_dnf(assignments)

    def _compute_assignments_array(self) -> jax.Array:
        """Returns the possible assignments in the environment in array form.

        Returns:
            jax.Array of shape (num_assignments, num_propositions) int32
        """
        assignments = -jnp.ones(
            (len(self.assignments), len(self.propositions)), dtype=jnp.int32
        )
        for i, assignment in enumerate(self.assignments):
            prop_indices = sorted(
                [self.propositions.index(p) for p in assignment], reverse=True
            )
            # padding
            prop_indices += [-1] * (len(self.propositions) - len(prop_indices))
            prop_indices = jnp.array(prop_indices, dtype=jnp.int32)
            assignments = assignments.at[i, :].set(prop_indices)
        return assignments

    @property
    @abstractmethod
    def assignments(self) -> list[Assignment]:
        """Returns the possible assignments as a list of Assignment objects."""
        pass

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    def unwrapped(self, state: Any) -> TEnvState:
        """Returns the unwrapped environment state."""
        return state

    @abstractmethod
    def get_renderer(
        self, params: TEnvParams, **kwargs
    ) -> "BaseRenderer[TObsFeatures, TResetOptions]":
        """Returns a renderer for the environment."""
        pass

    def plot_trajectories(
        self,
        trajs: TEnvState,
        lengths: jax.Array,
        params: TEnvParams,
        **plotting_kwargs,
    ) -> None:
        """Plots trajectories of environment states.

        Args:
            trajs: Batched EnvStates of shape (num_episodes, max_length, ...)
            lengths: Trajectory lengths (num_episodes,) int32
            params: Environment parameters
            plotting_kwargs: Additional keyword arguments for the plotting function
        """
        raise NotImplementedError(
            f"plot_trajectories not implemented for {self.name} environment."
        )
