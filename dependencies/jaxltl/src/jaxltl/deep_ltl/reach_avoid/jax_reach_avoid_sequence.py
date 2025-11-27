from dataclasses import KW_ONLY, field, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import (
    EpsilonType,
    ReachAvoidSequence,
)
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper


class JaxReachAvoidSequence(eqx.Module):
    """Jax representation of a reach-avoid sequence consisting of sets of assignments to reach and avoid."""

    # Each row consists of assignment indices with -1 padding. Epsilon transitions are
    # represented by an index of len(env.assignments) in the reach set.
    reach: jax.Array  # shape: (max_length, num_assignments)
    avoid: jax.Array  # shape: (max_length, num_assignments)

    # Defaults go after this line
    _: KW_ONLY

    # Indicates how often the last reach-avoid pair should be repeated. This is used to
    # specify long sequences (such as FG a) without increasing the size of the arrays.
    repeat_last: jax.Array = field(
        default_factory=lambda: jnp.ones((), dtype=jnp.int32)
    )
    # counts how often advanced in last step
    last_index: jax.Array = field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.int32)
    )

    def advance(self) -> "JaxReachAvoidSequence":
        """Advance the reach-avoid sequence by one step. Returns a new sequence, with
        the last step padded.
        """
        is_last_step = self.depth == 1
        should_repeat = jnp.logical_and(
            is_last_step, self.last_index + 1 < self.repeat_last
        )

        def _repeat_step():
            return replace(self, last_index=self.last_index + 1)

        def _advance_step():
            # advance arrays one step
            new_reach = jnp.roll(self.reach, -1, axis=0)
            new_avoid = jnp.roll(self.avoid, -1, axis=0)

            # pad the last row with -1s
            new_reach = new_reach.at[-1, :].set(-1)
            new_avoid = new_avoid.at[-1, :].set(-1)

            return JaxReachAvoidSequence(
                reach=new_reach,
                avoid=new_avoid,
                repeat_last=self.repeat_last,
                last_index=jnp.zeros((), dtype=jnp.int32),
            )

        seq = jax.lax.cond(
            should_repeat,
            _repeat_step,
            _advance_step,
        )
        return seq

    @property
    def depth(self) -> jax.Array:
        """Compute the depth of the sequence (number of non-padded steps)."""
        # Depth is determined by the assignment sequence
        padded_steps = self.reach[..., 0] == -1
        return jnp.sum(~padded_steps, axis=-1)

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[ReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxReachAvoidSequence":
        """Converts a mapping from LDBA states to lists of ReachAvoidSequences into a
        batched Jax reach-avoid sequence.

        Returns:
            JaxReachAvoidSequence: with shape
                reach: (num_states, max_num_seqs, max_length, num_assignments)
                avoid: (num_states, max_num_seqs, max_length, num_assignments)
        """

        max_seqs = max(len(seqs) for seqs in state_to_seqs.values())
        max_length = max(
            len(seq.reach_avoid) for seqs in state_to_seqs.values() for seq in seqs
        )
        num_states = len(state_to_seqs)
        # Use numpy arrays and then convert to jax arrays for efficiency
        reach = -np.ones(
            (num_states, max_seqs, max_length, len(env.assignments)),
            dtype=np.int32,
        )
        avoid = -np.ones_like(reach)
        for state, seqs in state_to_seqs.items():
            for seq_idx, seq in enumerate(seqs):
                for i, (r, a) in enumerate(seq.reach_avoid):
                    if isinstance(r, EpsilonType):
                        reach[state, seq_idx, i, 0] = len(env.assignments)
                    else:
                        for j, assignment in enumerate(r):
                            reach[state, seq_idx, i, j] = env.assignments.index(
                                assignment
                            )
                    for j, assignment in enumerate(a):
                        avoid[state, seq_idx, i, j] = env.assignments.index(assignment)
        return cls(
            reach=jnp.array(reach),
            avoid=jnp.array(avoid),
            repeat_last=jnp.ones((num_states, max_seqs), dtype=jnp.int32),
            last_index=jnp.zeros((num_states, max_seqs), dtype=jnp.int32),
        )
