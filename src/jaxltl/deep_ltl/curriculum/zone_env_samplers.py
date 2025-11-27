import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.sampling_utils import sample_assignments
from jaxltl.deep_ltl.curriculum.sequence_sampler import AssignmentSequenceSampler
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence


class ZoneReachAvoidSampler(AssignmentSequenceSampler):
    """Samples simple reach-avoid sequences.

    Note: we assume that the last assignment index corresponds to the empty assignment,
    which we do not sample.
    """

    depth: tuple[int, int]
    reach: tuple[int, int]
    avoid: tuple[int, int]

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        *,
        num_assignments: int,
        max_length: int,
    ):
        super().__init__(num_assignments, max_length)
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.depth = depth
        self.reach = reach
        self.avoid = avoid

    @eqx.filter_jit
    def sample(self, key: jax.Array) -> JaxReachAvoidSequence:
        key, depth_key = jax.random.split(key)
        depth = jax.random.randint(depth_key, (), self.depth[0], self.depth[1] + 1)

        # 1. Pre-allocate output arrays (filled with padding)
        reach_seq = -jnp.ones((self.max_length, self.num_assignments), dtype=jnp.int32)
        avoid_seq = -jnp.ones((self.max_length, self.num_assignments), dtype=jnp.int32)

        def body_fn(i, carry):
            """
            This function is executed for each step of the jax.lax.fori_loop.
            `i` is the loop index (from 0 to depth-1).
            `carry` holds the state: (key, last_reach_mask, reach_seq, avoid_seq)
            """
            key, last_reach_props_mask, reach_seq_carry, avoid_seq_carry = carry

            key, reach_key, avoid_key = jax.random.split(key, 3)

            # --- 1. Sample Reach Set ---
            reach_mask = sample_assignments(
                ~last_reach_props_mask, self.reach, reach_key
            )
            new_last_reach_props_mask = reach_mask

            # --- 2. Sample Avoid Set ---
            available_avoid_mask = ~reach_mask & ~last_reach_props_mask
            num_available_avoid = jnp.sum(available_avoid_mask)
            na_min_clamped = jnp.minimum(self.avoid[0], num_available_avoid)
            na_max_clamped = jnp.minimum(self.avoid[1], num_available_avoid)

            avoid_mask = sample_assignments(
                available_avoid_mask, (na_min_clamped, na_max_clamped), avoid_key
            )

            # --- 3. Update the output arrays at index `i` ---
            reach = jnp.nonzero(reach_mask, size=self.num_assignments, fill_value=-1)[0]
            reach = jnp.sort(reach, descending=True)
            avoid = jnp.nonzero(avoid_mask, size=self.num_assignments, fill_value=-1)[0]
            avoid = jnp.sort(avoid, descending=True)
            reach_seq_new = reach_seq_carry.at[i].set(reach)
            avoid_seq_new = avoid_seq_carry.at[i].set(avoid)

            new_carry = (key, new_last_reach_props_mask, reach_seq_new, avoid_seq_new)
            return new_carry

        # 2. Define the initial state for the loop
        initial_carry = (
            key,
            # initial last_reach_mask, without empty assignment
            jnp.zeros(self.num_assignments - 1, dtype=bool),
            reach_seq,  # initial (empty) reach_seq
            avoid_seq,  # initial (empty) avoid_seq
        )

        # 3. Run the loop from 0 up to (but not including) `depth`
        final_carry = jax.lax.fori_loop(0, depth, body_fn, initial_carry)

        # 4. Extract the final arrays from the carry
        _, _, final_reach_seq, final_avoid_seq = final_carry

        return JaxReachAvoidSequence(reach=final_reach_seq, avoid=final_avoid_seq)


class ZoneReachStaySampler(AssignmentSequenceSampler):
    """Samples reach-stay sequences."""

    num_stay: jax.Array  # int32, the number of timesteps to stay after reaching
    avoid: tuple[int, int]

    def __init__(
        self,
        num_stay: int,
        avoid: int | tuple[int, int],
        *,
        num_assignments: int,
        max_length: int,
    ):
        super().__init__(num_assignments, max_length)
        if max_length <= 1:
            raise ValueError(
                "max_length must be greater than 1 for reach-stay sequences."
            )
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.avoid = avoid
        self.num_stay = jnp.array(num_stay, dtype=jnp.int32)

    @eqx.filter_jit
    def sample(self, key: jax.Array) -> JaxReachAvoidSequence:
        reach_seq = -jnp.ones((self.max_length, self.num_assignments), dtype=jnp.int32)
        avoid_seq = -jnp.ones((self.max_length, self.num_assignments), dtype=jnp.int32)

        # 1. Sample proposition to reach
        reach_key, avoid_key = jax.random.split(key)
        reach_prop = jax.random.randint(reach_key, (), 0, self.num_assignments - 1)

        # 2. Sample avoid set
        available_avoid_mask = (
            jnp.ones(self.num_assignments - 1, dtype=bool).at[reach_prop].set(False)
        )
        avoid_mask = sample_assignments(available_avoid_mask, self.avoid, avoid_key)
        avoid = jnp.nonzero(avoid_mask, size=self.num_assignments, fill_value=-1)[0]
        avoid = jnp.sort(avoid, descending=True)

        # 3. Set epsilon transition with avoid set
        reach_seq = reach_seq.at[0, 0].set(self.num_assignments)  # epsilon transition
        avoid_seq = avoid_seq.at[0].set(avoid)

        # 4. Build avoid everything after reaching
        avoid = jnp.arange(self.num_assignments, dtype=jnp.int32)
        avoid = avoid.at[reach_prop].set(-1)  # exclude reached proposition
        avoid = jnp.sort(avoid, descending=True)

        # 5. Fill in 2 stay steps
        for i in range(1, min(3, self.max_length)):
            reach_seq = reach_seq.at[i, 0].set(reach_prop)
            avoid_seq = avoid_seq.at[i].set(avoid)

        return JaxReachAvoidSequence(
            reach=reach_seq, avoid=avoid_seq, repeat_last=self.num_stay
        )
