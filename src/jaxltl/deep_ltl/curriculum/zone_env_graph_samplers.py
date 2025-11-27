import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxltl.deep_ltl.curriculum.graph_sequence_sampler import GraphSequenceSampler
from jaxltl.deep_ltl.curriculum.sampling_utils import sample_propositions
from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import EPSILON
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import MultiOrNode, Node, NotNode, VarNode


def _create_or_graph(nodes: list[VarNode]) -> Node | None:
    """Creates a graph from a list of VarNodes."""
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    return MultiOrNode(nodes)


@functools.lru_cache(maxsize=1024)
def _compute_satisfying_assignments(
    graph: Node | None, all_assignments: tuple[Assignment, ...]
) -> frozenset[Assignment]:
    """Computes the set of assignments that satisfy a given boolean formula."""
    if graph is None:
        return frozenset()
    return frozenset(a for a in all_assignments if graph.eval(a))


class GraphZoneReachAvoidSampler(GraphSequenceSampler):
    """Samples simple reach-avoid sequences by sampling boolean formulae."""

    depth: tuple[int, int]
    reach: tuple[int, int]
    avoid: tuple[int, int]

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        *,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_length: int,
        max_nodes: int,
        max_edges: int,
    ):
        super().__init__(propositions, assignments, max_length, max_nodes, max_edges)
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.depth = depth
        self.reach = reach
        self.avoid = avoid

    def sample_graph(self, key: jax.Array) -> GraphReachAvoidSequence:
        key, depth_key = jax.random.split(key)
        depth = jax.random.randint(depth_key, (), self.depth[0], self.depth[1] + 1)

        reach_avoid_assignments = []
        reach_avoid_graphs = []

        available_props_mask = jnp.ones(len(self.propositions), dtype=bool)

        while (
            len(reach_avoid_graphs) < depth
            and jnp.sum(available_props_mask) >= self.reach[0]
        ):
            key, reach_key, avoid_key = jax.random.split(key, 3)

            # 1. Sample Reach Formula
            reach_props_mask = sample_propositions(
                reach_key, available_props_mask, self.reach
            )
            reach_prop_indices = np.where(reach_props_mask)[0]
            reach_nodes = [VarNode(self.propositions[i]) for i in reach_prop_indices]
            reach_graph = _create_or_graph(reach_nodes)

            available_props_mask &= ~reach_props_mask

            # 2. Sample Avoid Formula
            avoid_props_mask = sample_propositions(
                avoid_key, available_props_mask, self.avoid
            )
            avoid_prop_indices = np.where(avoid_props_mask)[0]
            avoid_nodes = [VarNode(self.propositions[i]) for i in avoid_prop_indices]
            avoid_graph = _create_or_graph(avoid_nodes)

            available_props_mask &= ~avoid_props_mask

            # 3. Compute satisfying assignments
            reach_assigns = _compute_satisfying_assignments(
                reach_graph, self.assignments
            )
            avoid_assigns = _compute_satisfying_assignments(
                avoid_graph, self.assignments
            )

            reach_avoid_graphs.append((reach_graph, avoid_graph))
            reach_avoid_assignments.append((reach_assigns, avoid_assigns))

        # Pad to max_length
        num_padding = self.max_length - len(reach_avoid_graphs)
        if num_padding > 0:
            padding_assignments = (frozenset(), frozenset())
            padding_graphs = (None, None)
            reach_avoid_assignments.extend([padding_assignments] * num_padding)
            reach_avoid_graphs.extend([padding_graphs] * num_padding)

        return GraphReachAvoidSequence(reach_avoid_assignments, reach_avoid_graphs)


class GraphZoneReachStaySampler(GraphSequenceSampler):
    """Samples reach-stay sequences by sampling boolean formulae."""

    num_stay: int
    avoid: tuple[int, int]

    def __init__(
        self,
        num_stay: int,
        avoid: int | tuple[int, int],
        *,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_length: int,
        max_nodes: int,
        max_edges: int,
    ):
        super().__init__(propositions, assignments, max_length, max_nodes, max_edges)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.avoid = avoid
        self.num_stay = num_stay

    def sample_graph(self, key: jax.Array) -> GraphReachAvoidSequence:
        reach_key, avoid_key = jax.random.split(key)

        # 1. Sample proposition to reach
        reach_prop_idx = jax.random.randint(reach_key, (), 0, len(self.propositions))
        reach_prop_name = self.propositions[reach_prop_idx]
        reach_graph = VarNode(reach_prop_name)

        # 2. Sample initial avoid set
        available_avoid_mask = jnp.ones(len(self.propositions), dtype=bool)
        available_avoid_mask = available_avoid_mask.at[reach_prop_idx].set(False)
        avoid_mask = sample_propositions(avoid_key, available_avoid_mask, self.avoid)
        avoid_indices = np.where(avoid_mask)[0]
        avoid_nodes = [VarNode(self.propositions[i]) for i in avoid_indices]
        initial_avoid_graph = _create_or_graph(avoid_nodes)

        # 3. Create "stay" avoid graph
        stay_avoid_graph = NotNode(reach_graph)

        # 4. Compute all assignment sets
        initial_avoid_assigns = _compute_satisfying_assignments(
            initial_avoid_graph, self.assignments
        )
        reach_assigns = _compute_satisfying_assignments(reach_graph, self.assignments)
        stay_avoid_assigns = _compute_satisfying_assignments(
            stay_avoid_graph, self.assignments
        )

        # 5. Build the sequence
        reach_avoid_graphs = [
            (EPSILON, initial_avoid_graph),
            (reach_graph, stay_avoid_graph),
        ]
        reach_avoid_assignments = [
            (EPSILON, initial_avoid_assigns),
            (reach_assigns, stay_avoid_assigns),
        ]

        # Add a second stay step if there is space
        if self.max_length > 2:
            reach_avoid_graphs.append((reach_graph, stay_avoid_graph))
            reach_avoid_assignments.append((reach_assigns, stay_avoid_assigns))

        # Pad to max_length
        num_padding = self.max_length - len(reach_avoid_graphs)
        if num_padding > 0:
            padding_assignments = (frozenset(), frozenset())
            padding_graphs = (None, None)
            reach_avoid_assignments.extend([padding_assignments] * num_padding)
            reach_avoid_graphs.extend([padding_graphs] * num_padding)

        return GraphReachAvoidSequence(
            reach_avoid_assignments, reach_avoid_graphs, repeat_last=self.num_stay
        )
