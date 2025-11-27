from collections.abc import Sequence

import jax

from jaxltl.deep_ltl.curriculum.sequence_sampler import SequenceSampler
from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxltl.ltl.logic.assignment import Assignment


class GraphSequenceSampler(SequenceSampler):
    """Base class for graph-based sequence samplers."""

    propositions: tuple[str, ...]
    assignments: tuple[Assignment, ...]
    max_length: int
    max_nodes: int
    max_edges: int

    def __init__(
        self,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_length: int,
        max_nodes: int,
        max_edges: int,
    ):
        self.propositions = tuple(propositions)
        self.assignments = tuple(assignments)
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    def sample(self, key: jax.Array) -> JaxGraphReachAvoidSequence:
        graph_seq = self.sample_graph(key)
        return JaxGraphReachAvoidSequence.from_seq(
            graph_seq,
            self.propositions,
            self.assignments,
            self.max_nodes,
            self.max_edges,
        )

    def sample_graph(self, key: jax.Array) -> GraphReachAvoidSequence:
        raise NotImplementedError
