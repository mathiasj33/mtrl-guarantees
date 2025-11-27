"""Module for sequence sampling strategies in DeepLTL."""

from abc import abstractmethod

import equinox as eqx
import jax

from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence


class SequenceSampler(eqx.Module):
    """Base class for sequence sampling strategies."""

    @abstractmethod
    def sample(self, key: jax.Array) -> JaxReachAvoidSequence:
        """Sample a reach-avoid sequence."""
        pass


class AssignmentSequenceSampler(SequenceSampler):
    """Base class for assignment-based sequence samplers."""

    num_assignments: int
    max_length: int

    def __init__(self, num_assignments: int, max_length: int):
        self.num_assignments = num_assignments
        self.max_length = max_length
