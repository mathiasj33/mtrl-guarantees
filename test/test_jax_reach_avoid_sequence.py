from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import EPSILON, ReachAvoidSequence
from jaxltl.environments.environment import Environment
from jaxltl.ltl.logic.assignment import Assignment


def make_set(*assignments: set[str]) -> frozenset[Assignment]:
    return frozenset(Assignment(frozenset(a)) for a in assignments)


class MockEnv(Environment):
    def __init__(self):
        super().__init__(
            default_params=None, propositions=("red", "green", "purple", "yellow")
        )

    @property
    @override
    def assignments(self):
        return [
            Assignment(frozenset({"red"})),
            Assignment(frozenset({"green"})),
            Assignment(frozenset({"purple"})),
            Assignment(frozenset({"yellow"})),
            Assignment(frozenset({"red", "yellow"})),
            Assignment(frozenset()),
        ]

    def _reset(self):
        pass

    def _cheap_reset(self):
        pass

    def _step(self):
        pass

    def _compute_obs(self):
        pass

    def compute_propositions(self):
        pass

    def _observation_space(self):
        pass

    def _action_space(self):
        pass

    def get_renderer(self):
        pass


def test_from_state_to_seq():
    mock_env = MockEnv()
    ras = ReachAvoidSequence(
        reach_avoid=[
            (make_set({"green"}, {"red"}), frozenset()),
            (make_set({"purple"}), make_set({"green"}, {"yellow"})),
            (make_set({"yellow", "red"}, {"red"}), make_set({"yellow"})),
            (EPSILON, make_set({"yellow"})),
            (
                make_set({"green"}),
                make_set({"red"}, {"purple"}, {"yellow"}, {"red", "yellow"}, set()),
            ),
        ]
    )
    state_to_seq = {0: [ras]}
    jax_ras = JaxReachAvoidSequence.from_state_to_seqs(state_to_seq, mock_env)
    # remove batch dimension
    jax_ras = jax.tree.map(lambda x: x[0, 0], jax_ras)

    expected_reach = jnp.array(
        [
            [1, 0, -1, -1, -1, -1],
            [2, -1, -1, -1, -1, -1],
            [4, 0, -1, -1, -1, -1],
            [6, -1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )

    expected_avoid = jnp.array(
        [
            [-1, -1, -1, -1, -1, -1],
            [1, 3, -1, -1, -1, -1],
            [3, -1, -1, -1, -1, -1],
            [3, -1, -1, -1, -1, -1],
            [0, 2, 3, 4, 5, -1],
        ],
        dtype=jnp.int32,
    )

    expected_jax_ras = JaxReachAvoidSequence(expected_reach, expected_avoid)
    assert_ragged_set_equal(jax_ras.reach, expected_jax_ras.reach)
    assert_ragged_set_equal(jax_ras.avoid, expected_jax_ras.avoid)


def assert_ragged_set_equal(actual, expected, pad_val=-1):
    """
    Asserts that two dense arrays representing ragged sets are equal.

    This checks two things:
    1. The padding structure (locations of pad_val) is identical.
    2. The non-padded elements in each row are the same,
       ignoring their order.
    """
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)

    npt.assert_equal(
        actual_np.shape, expected_np.shape, "Arrays must have the same shape."
    )

    actual_mask = actual_np != pad_val
    expected_mask = expected_np != pad_val
    npt.assert_array_equal(
        actual_mask, expected_mask, "Padding structure (mask) must be identical."
    )

    # 3. Check that the non-padded elements in each row are a
    #    set-equivalent match (by sorting them)
    for i in range(actual_np.shape[0]):
        # Extract only the valid (non-padded) elements from this row
        actual_valid = actual_np[i][actual_mask[i]]
        expected_valid = expected_np[i][expected_mask[i]]

        # Sort the valid elements to make the comparison order-invariant
        npt.assert_array_equal(
            np.sort(actual_valid),
            np.sort(expected_valid),
            f"Row {i} non-padded elements do not match (order-invariant): {actual_np[i]} vs {expected_np[i]}",
        )
