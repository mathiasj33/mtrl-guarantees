from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from jaxltl import eqx_utils
from jaxltl.deep_ltl.curriculum.curriculum import JaxReachAvoidSequence
from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxltl.deep_ltl.wrappers.curriculum_wrapper import SequenceObservation
from jaxltl.environments.environment import EnvObservation, EnvParams, EnvTransition
from jaxltl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxltl.environments.wrappers.wrapper import EnvWrapper, WrapperState
from jaxltl.ltl.automata.jax_ldba import JaxLDBA
from jaxltl.rl.actor_critic import ActorCritic


class EvalState(NamedTuple):
    """State to keep track of during evaluation."""

    ldba_state: jax.Array  # int, current LDBA state (batched)
    seq: JaxReachAvoidSequence  # current reach-avoid sequence (batched)
    metric: jax.Array  # (num_episodes,) success/accepting visits and total steps
    returns: jax.Array  # (num_episodes,) discounted returns
    lengths: jax.Array  # (num_episodes,) lengths of episodes
    completed: jax.Array  # (num_episodes,) whether episode is completed


class Evaluator(eqx.Module):
    """Implements model evaluation. Keeps track of the LDBA state and selected
    reach-avoid sequence. Keeps track of success rate (for finite formulas) or visits to
    accepting states (for infinite formulas).

    Note: Assumes that the underlying environment is already vectorized.
    This is for efficiency reasons, since the ActorCritic model already expects batched inputs.
    """

    num_episodes: int  # Number of evaluation episodes. We run each episode in parallel.
    discount: float  # Discount factor for returns.

    def __init__(self, num_episodes: int, discount: float):
        self.num_episodes = num_episodes
        self.discount = discount

    @eqx.filter_jit
    def eval(
        self,
        model: ActorCritic,
        deterministic: bool,
        env: VectorizeWrapper,
        env_params: EnvParams,
        ldba: JaxLDBA,
        batched_seqs: JaxReachAvoidSequence,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, PyTree]:
        """Evaluate the model on the environment with the given LDBA and batched sequences.

        Returns:
            metric: jax.Array of shape (num_episodes,) with success/accepting visits
            returns: jax.Array of shape (num_episodes,) with discounted returns
            lengths: jax.Array of shape (num_episodes,) with lengths of episodes
            trajs: PyTree of shape (num_episodes, max_length, ...) with env states
        """

        def rollout_cond(
            carry: tuple[
                WrapperState, PyTree, jax.Array, EvalState, PyTree, jax.Array, jax.Array
            ],
        ) -> jax.Array:
            env_state, obsv, props, eval_state, trajs, index, key = carry
            return jnp.sum(eval_state.completed).astype(jnp.int32) < self.num_episodes

        def rollout_step(
            carry: tuple[
                WrapperState, PyTree, jax.Array, EvalState, PyTree, jax.Array, jax.Array
            ],
        ) -> tuple[
            WrapperState, PyTree, jax.Array, EvalState, PyTree, jax.Array, jax.Array
        ]:
            env_state, obsv, props, eval_state, trajs, index, key = carry
            # select action
            assignment_index = jax.vmap(env.map_assignment_to_index)(props)
            eps_enabled = jax.vmap(self._is_epsilon_enabled, in_axes=(None, 0, 0))(
                env, eval_state.seq, assignment_index
            )
            seq_obsv = SequenceObservation.from_obs(obsv, eval_state.seq, eps_enabled)
            pi = model.get_action(seq_obsv)
            if deterministic:
                action = pi.mode()
            else:
                key, sample_key = jax.random.split(key)
                action = pi.sample(seed=sample_key)
            env_action, epsilon_action = action

            # step env
            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.num_episodes)
            env_transition = env.step(step_key, env_state, env_action, env_params)
            eps_transition = self._epsilon_step(env_state, obsv, props)
            transition = eqx_utils.pytree_where(
                epsilon_action.astype(jnp.bool), eps_transition, env_transition
            )

            # record trajectory
            trajs = jax.tree.map(
                lambda x, y: x.at[:, index + 1].set(y),
                trajs,
                transition.state.state,
            )

            # update LDBA state
            # (num_envs,) int32
            assignments = jax.vmap(env.map_assignment_to_index)(transition.propositions)
            next_ldba_state, is_accepting = jax.vmap(ldba.get_next_state)(
                eval_state.ldba_state, assignments
            )
            next_eps_state = jax.vmap(ldba.get_next_epsilon_state)(
                eval_state.ldba_state
            )
            next_ldba_state = jnp.where(
                epsilon_action.astype(jnp.bool), next_eps_state, next_ldba_state
            )
            is_accepting = jnp.where(  # epsilon transitions cannot be accepting
                epsilon_action.astype(jnp.bool), False, is_accepting
            )
            is_sink = jax.vmap(ldba.is_sink_state)(next_ldba_state)

            # choose new sequences based on updated LDBA state
            needs_update = next_ldba_state != eval_state.ldba_state
            new_seqs = self._choose_sequences(
                model,
                next_ldba_state,
                batched_seqs,
                transition.observation,
            )
            seq = eqx_utils.pytree_where(needs_update, new_seqs, eval_state.seq)

            # update metrics
            rewards = is_accepting.astype(jnp.int32)
            rewards = jnp.where(eval_state.completed, 0, rewards)
            metrics = eval_state.metric + rewards
            returns = eval_state.returns + jnp.power(self.discount, index) * rewards
            lengths = eval_state.lengths + jnp.where(eval_state.completed, 0, 1)

            # update completed
            new_completed = jnp.logical_or(
                transition.done, jnp.logical_and(ldba.finite, is_accepting)
            )
            new_completed = jnp.logical_or(new_completed, is_sink)
            completed = jnp.logical_or(eval_state.completed, new_completed)

            new_eval_state = EvalState(
                ldba_state=next_ldba_state,
                seq=seq,
                metric=metrics,
                returns=returns,
                lengths=lengths,
                completed=completed,
            )
            return (
                transition.state,
                transition.observation,
                transition.propositions,
                new_eval_state,
                trajs,
                index + 1,
                key,
            )

        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.num_episodes)
        env_state, obsv = env.reset(reset_keys, None, env_params, None)
        props: jax.Array = eqx.filter_vmap(env.compute_propositions)(
            env_state, env_params
        )
        max_length = env_params.max_steps_in_episode
        trajs = jax.tree.map(
            lambda x: jnp.zeros(
                (self.num_episodes, max_length + 1) + x.shape[1:], dtype=x.dtype
            ),
            env_state.state,
        )
        trajs = jax.tree.map(lambda x, y: x.at[:, 0].set(y), trajs, env_state.state)
        index = jnp.zeros((), dtype=jnp.int32)
        ldba_state = jnp.full((self.num_episodes,), ldba.initial_state, dtype=jnp.int32)
        seqs = self._choose_sequences(model, ldba_state, batched_seqs, obsv)
        state = EvalState(
            ldba_state=ldba_state,
            seq=seqs,
            metric=jnp.zeros((self.num_episodes,), dtype=jnp.int32),
            returns=jnp.zeros((self.num_episodes,), dtype=jnp.float32),
            lengths=jnp.zeros((self.num_episodes,), dtype=jnp.int32),
            completed=jnp.zeros((self.num_episodes,), dtype=bool),
        )
        final = jax.lax.while_loop(
            rollout_cond,
            rollout_step,
            (env_state, obsv, props, state, trajs, index, key),
        )
        _, _, _, state, trajs, _, _ = final
        return state.metric, state.returns, state.lengths, trajs

    def _epsilon_step(
        self,
        env_state: WrapperState,
        obsv: EnvObservation,
        propositions: jax.Array,
    ) -> EnvTransition:
        """Do-nothing transition corresponding to an epsilon action."""
        return EnvTransition(
            state=env_state,
            observation=obsv,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=jnp.bool),
            truncated=jnp.zeros((), dtype=jnp.bool),
            terminal_observation=obsv,
            propositions=propositions,
            info={},
        )

    @eqx.filter_jit
    def _choose_sequences(
        self,
        model: ActorCritic,
        ldba_state: jax.Array,
        batched_seqs: JaxReachAvoidSequence,
        obsv: EnvObservation,
    ) -> JaxReachAvoidSequence:
        """Selects the best reach-avoid sequence for each environment based on the
        current observation and LDBA state."""

        def choose_sequence_for_env(
            ldba_state: jax.Array, obs: EnvObservation
        ) -> JaxReachAvoidSequence:
            # ldba_state: int
            # obs: EnvObservation
            if type(batched_seqs) is JaxGraphReachAvoidSequence:
                state_seqs = JaxGraphReachAvoidSequence(
                    reach=batched_seqs.reach[ldba_state],
                    avoid=batched_seqs.avoid[ldba_state],
                    reach_graphs=jax.tree.map(
                        lambda x: x[ldba_state], batched_seqs.reach_graphs
                    ),
                    avoid_graphs=jax.tree.map(
                        lambda x: x[ldba_state], batched_seqs.avoid_graphs
                    ),
                    repeat_last=batched_seqs.repeat_last[ldba_state],
                    last_index=batched_seqs.last_index[ldba_state],
                )
            else:
                state_seqs = JaxReachAvoidSequence(
                    reach=batched_seqs.reach[ldba_state],
                    avoid=batched_seqs.avoid[ldba_state],
                    repeat_last=batched_seqs.repeat_last[ldba_state],
                    last_index=batched_seqs.last_index[ldba_state],
                )
            num_seqs = state_seqs.reach.shape[0]
            batched_obs = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None, ...], (num_seqs,) + x.shape), obs
            )
            batched_seq_obs = SequenceObservation.from_obs(
                batched_obs,
                state_seqs,
                epsilon_enabled=jnp.ones((num_seqs,), dtype=bool),
            )  # epsilon_enabled is irrelevant for the critic
            scores = model.get_value(batched_seq_obs)  # (num_seqs,)
            padded = state_seqs.reach[:, 0, 0] == -1
            scores = jnp.where(padded, -jnp.inf, scores)
            best_index = jnp.argmax(scores)
            return jax.tree.map(lambda x: x[best_index], state_seqs)

        batched_choose_sequence = jax.vmap(choose_sequence_for_env, in_axes=(0, 0))
        return batched_choose_sequence(ldba_state, obsv)

    def _is_epsilon_enabled(
        self, env: EnvWrapper, seq: JaxReachAvoidSequence, assignment_index: jax.Array
    ) -> jax.Array:
        """Returns a boolean indicating if an epsilon action can be taken. This is only
        true if the current step in the reach-avoid sequence is an epsilon transition,
        and the current environment assignment does not violate the next avoid set.
        """
        is_epsilon = seq.reach[0, 0] == len(env._env.assignments)
        is_valid = jnp.logical_or(
            seq.depth <= 1, jnp.all(seq.avoid[1] != assignment_index)
        )
        return jnp.logical_and(is_epsilon, is_valid)
