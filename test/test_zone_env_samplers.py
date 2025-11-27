import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.zone_env_samplers import (
    ZoneReachAvoidSampler,
    ZoneReachStaySampler,
)


def test_reach_avoid_sampler():
    sampler = ZoneReachAvoidSampler(
        depth=(3, 8),
        reach=(1, 3),
        avoid=(0, 2),
        num_assignments=5,
        max_length=9,
    )
    key = jax.random.key(0)
    for j in range(100):
        key, subkey = jax.random.split(key)
        seq = sampler.sample(subkey)

        if j < 5:
            print(
                f"\nSampled Reach-Avoid Sequence ({j}):\nReach:\n{jax.device_get(seq.reach)}\nAvoid:\n{jax.device_get(seq.avoid)}"
            )

        # Check shapes
        assert seq.reach.shape == (sampler.max_length, sampler.num_assignments)
        assert seq.avoid.shape == (sampler.max_length, sampler.num_assignments)

        # Check depth constraints
        assert sampler.depth[0] <= seq.depth <= sampler.depth[1]

        for i in range(seq.depth):
            # Check reach set size constraints
            reach_set = seq.reach[i]
            num_reach = jnp.sum(reach_set != -1)
            assert sampler.reach[0] <= num_reach <= sampler.reach[1]

            # Check avoid set size constraints
            avoid_set = seq.avoid[i]
            num_avoid = jnp.sum(avoid_set != -1)
            assert sampler.avoid[0] <= num_avoid <= sampler.avoid[1]

            # Check reach-avoid disjointness
            assert len(set(reach_set.tolist()) & set(avoid_set.tolist()) - {-1}) == 0

            # Check reach-last reach disjointness
            if i > 0:
                last_reach_set = seq.reach[i - 1]
                assert (
                    len(set(reach_set.tolist()) & set(last_reach_set.tolist()) - {-1})
                    == 0
                )

            # Check that empty assignment is not included
            empty_assignment = sampler.num_assignments - 1
            assert empty_assignment not in reach_set.tolist()
            assert empty_assignment not in avoid_set.tolist()

        for i in range(seq.depth, seq.reach.shape[0]):
            # Check padding
            assert jnp.all(seq.reach[i] == -1)
            assert jnp.all(seq.avoid[i] == -1)


def test_reach_stay_sampler():
    sampler = ZoneReachStaySampler(
        num_stay=30, avoid=(0, 2), num_assignments=5, max_length=3
    )
    key = jax.random.key(0)
    for j in range(100):
        key, subkey = jax.random.split(key)
        seq = sampler.sample(subkey)

        if j < 5:
            print(
                f"\nSampled Reach-Stay Sequence ({j}):\nReach:\n{jax.device_get(seq.reach)}\nAvoid:\n{jax.device_get(seq.avoid)}"
            )

        # Check shapes
        assert seq.reach.shape == (sampler.max_length, sampler.num_assignments)
        assert seq.avoid.shape == (sampler.max_length, sampler.num_assignments)

        # Check first transition is epsilon transition
        reach_first = seq.reach[0]
        assert reach_first[0] == sampler.num_assignments

        # Check avoid set at first step
        avoid_first = seq.avoid[0]
        num_avoid_first = jnp.sum(avoid_first != -1)
        assert sampler.avoid[0] <= num_avoid_first <= sampler.avoid[1]

        # Check reach-stay steps
        reach_prop = seq.reach[1][0]
        assert 0 <= reach_prop < sampler.num_assignments - 1
        for i in range(1, 3):
            reach_step = seq.reach[i]
            avoid_step = seq.avoid[i]

            # Check reach proposition is maintained
            assert reach_step[0] == reach_prop

            # Check avoid set excludes reached proposition
            assert set(avoid_step.tolist()) - {-1} == set(
                range(sampler.num_assignments)
            ) - {int(reach_prop)}
