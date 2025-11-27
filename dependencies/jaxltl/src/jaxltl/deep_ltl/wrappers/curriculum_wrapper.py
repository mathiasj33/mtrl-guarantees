from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.curriculum import Curriculum, JaxReachAvoidSequence
from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper
from jaxltl.environments.wrappers.wrapper import WrapperState


class CurriculumState[TObsFeatures: NamedTuple](WrapperState):
    """State for CurriculumWrapper."""

    seq: JaxReachAvoidSequence  # current reach-avoid sequence

    # Curriculum-related state
    curriculum_stage: jax.Array  # current stage in the curriculum
    last_returns: jax.Array  # shape (N,), returns from last N episodes
    returns_index: jax.Array  # int, index to write next return into last_returns
    # int, number of completed episodes in the current stage
    current_stage_episodes: jax.Array

    # Epsilon-related state
    obs: EnvObservation[TObsFeatures]  # last observation
    propositions: jax.Array  # last propositions
    info: dict  # last info


class SequenceObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation returned by CurriculumWrapper."""

    seq: JaxReachAvoidSequence
    epsilon_mask: jax.Array  # bool, indicates if epsilon transition can be taken

    @classmethod
    def from_obs(
        cls,
        obs: EnvObservation[TObsFeatures],
        seq: JaxReachAvoidSequence,
        epsilon_enabled: jax.Array,
    ):
        return cls(features=obs.features, seq=seq, epsilon_mask=epsilon_enabled)


class CurriculumWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]):
    """A wrapper that adds reach-avoid sequences sampled from a curriculum to the environment."""

    curriculum: Curriculum
    episode_window: int  # number of episodes to consider for average return

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, TResetOptions]
        ),
        curriculum: Curriculum,
        episode_window: int,
    ):
        super().__init__(env)
        self.curriculum = curriculum
        self.episode_window = episode_window

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: CurriculumState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[CurriculumState, SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        re_state, obs = super().reset(reset_key, state, params, options)
        propositions = self._env.compute_propositions(re_state, params)
        state = self._wrap_reset_state(state, re_state, obs, propositions, sample_key)
        assignment = self._env.map_assignment_to_index(propositions)
        return state, SequenceObservation.from_obs(
            obs, state.seq, self._is_epsilon_enabled(state.seq, assignment)
        )

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: CurriculumState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[CurriculumState, SequenceObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)
        re_state, obs = super().cheap_reset(reset_key, state, params, options)
        propositions = self._env.compute_propositions(re_state, params)
        state = self._wrap_reset_state(state, re_state, obs, propositions, sample_key)
        assignment = self._env.map_assignment_to_index(propositions)
        return state, SequenceObservation.from_obs(
            obs, state.seq, self._is_epsilon_enabled(state.seq, assignment)
        )

    def _wrap_reset_state(
        self,
        state: CurriculumState | None,
        re_state: WrapperState,
        obs: EnvObservation[TObsFeatures],
        propositions: jax.Array,
        key: jax.Array,
    ) -> CurriculumState:
        if state is None:
            stage = jnp.zeros((), dtype=jnp.int32)
            return CurriculumState(
                state=re_state,
                seq=self.curriculum.sample(stage, key),
                curriculum_stage=stage,
                last_returns=jnp.zeros((self.episode_window,), dtype=jnp.float32),
                returns_index=jnp.zeros((), dtype=jnp.int32),
                current_stage_episodes=jnp.zeros((), dtype=jnp.int32),
                obs=obs,
                propositions=propositions,
                info={},
            )
        threshold = self.curriculum.threshold(state.curriculum_stage)
        avg_return = jax.lax.cond(
            state.current_stage_episodes < self.episode_window,
            lambda: -jnp.inf,
            lambda: jnp.mean(state.last_returns),
        )
        change_stage = avg_return >= threshold
        stage = jax.lax.cond(
            change_stage,
            lambda: state.curriculum_stage + 1,
            lambda: state.curriculum_stage,
        )
        last_returns = jax.lax.cond(
            change_stage,
            lambda: jnp.zeros((self.episode_window,), dtype=jnp.float32),
            lambda: state.last_returns,
        )
        current_stage_episodes = jax.lax.cond(
            change_stage,
            lambda: jnp.zeros((), dtype=jnp.int32),
            lambda: state.current_stage_episodes,
        )
        returns_index = jax.lax.cond(
            change_stage,
            lambda: jnp.zeros((), dtype=jnp.int32),
            lambda: state.returns_index,
        )
        seq = self.curriculum.sample(stage, key)
        return CurriculumState(
            state=re_state,
            seq=seq,
            curriculum_stage=stage,
            last_returns=last_returns,
            returns_index=returns_index,
            current_stage_episodes=current_stage_episodes,
            obs=obs,
            propositions=propositions,
            info={},
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: CurriculumState,
        action: tuple[jax.Array, jax.Array],  # (env_action, epsilon (int32))
        params: TEnvParams,
    ) -> EnvTransition[CurriculumState, TObsFeatures]:
        env_action, epsilon_action = action
        env_transition = super().step(key, state, env_action, params)
        transition = jax.lax.cond(
            epsilon_action.astype(jnp.bool),
            lambda: self._epsilon_step(state),
            lambda: env_transition,
        )
        seq = jax.lax.cond(
            epsilon_action.astype(jnp.bool),
            lambda: state.seq.advance(),
            lambda: state.seq,
        )
        seq = cast(JaxReachAvoidSequence, seq)
        reach = seq.reach[0]  # (num_assignments,)
        avoid = seq.avoid[0]
        assignment = self._env.map_assignment_to_index(transition.propositions)
        avoided = jnp.logical_not(jnp.any(avoid == assignment))
        reached = jnp.logical_and(jnp.any(reach == assignment), avoided)
        seq = jax.lax.cond(reached, lambda: seq.advance(), lambda: seq)
        reached_end = jnp.all(seq.reach[0] == -1)  # Check if reached is just padding
        reward = jax.lax.cond(
            reached_end,
            lambda: 1.0,
            lambda: jax.lax.cond(avoided, lambda: 0.0, lambda: -1.0),
        )
        terminated = jnp.logical_or(reached_end, ~avoided)
        new_returns = state.last_returns.at[state.returns_index].set(
            jax.nn.relu(reward)  # binary success indicator
        )
        last_returns = jnp.where(terminated, new_returns, state.last_returns)
        returns_index = jnp.where(
            terminated,
            (state.returns_index + 1) % self.episode_window,
            state.returns_index,
        )
        num_episodes = state.current_stage_episodes + terminated.astype(jnp.int32)
        new_state = CurriculumState(
            state=transition.state,
            seq=seq,
            curriculum_stage=state.curriculum_stage,
            last_returns=last_returns,
            returns_index=returns_index,
            current_stage_episodes=num_episodes,
            obs=transition.observation,
            propositions=transition.propositions,
            info=transition.info,
        )
        return EnvTransition(
            state=new_state,
            observation=SequenceObservation.from_obs(
                transition.observation,
                new_state.seq,
                self._is_epsilon_enabled(new_state.seq, assignment),
            ),
            reward=reward,
            terminated=jnp.logical_or(transition.terminated, terminated),
            truncated=transition.truncated,
            terminal_observation=SequenceObservation.from_obs(
                transition.terminal_observation,
                new_state.seq,
                self._is_epsilon_enabled(new_state.seq, assignment),
            ),
            propositions=transition.propositions,
            info=transition.info,
        )

    def _epsilon_step(
        self, state: CurriculumState
    ) -> EnvTransition[WrapperState, TObsFeatures]:
        """Transition corresponding to an epsilon action."""
        return EnvTransition(
            state=state.unwrapped(),
            observation=state.obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=jnp.bool),
            truncated=jnp.zeros((), dtype=jnp.bool),
            terminal_observation=state.obs,
            propositions=state.propositions,
            info=state.info,
        )

    def _is_epsilon_enabled(
        self, seq: JaxReachAvoidSequence, assignment_index: jax.Array
    ) -> jax.Array:
        """Returns a boolean indicating if an epsilon action can be taken. This is only
        true if the current step in the reach-avoid sequence is an epsilon transition,
        and the current environment assignment does not violate the next avoid set.
        """
        is_epsilon = seq.reach[0, 0] == len(self._env.assignments)
        is_valid = jnp.logical_or(
            seq.depth <= 1, jnp.all(seq.avoid[1] != assignment_index)
        )
        return jnp.logical_and(is_epsilon, is_valid)
