from collections.abc import Sequence
from dataclasses import replace
from typing import TypedDict, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import EpsilonType
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import (
    AndNode,
    EmptyNode,
    MultiAndNode,
    MultiOrNode,
    Node,
    NotNode,
    OrNode,
    VarNode,
)

# Define integer constants for node types
NODE_TYPE_AND = 0
NODE_TYPE_OR = 1
NODE_TYPE_NOT = 2
NODE_TYPE_EMPTY = 3
NODE_TYPE_EPSILON = 4


class NodeData(TypedDict):
    prop_idx: jax.Array
    type_idx: jax.Array
    mask: jax.Array


class EdgeData(TypedDict):
    mask: jax.Array


def _roll_graphs(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Rolls a single sequence of graphs by one step."""
    # This function is only designed to work on a single, unbatched sequence.

    # Infer max_nodes and max_edges from the total size and sequence length.
    nodes = cast(NodeData, graphs.nodes)
    edges = cast(EdgeData, graphs.edges)
    senders = cast(jax.Array, graphs.senders)
    receivers = cast(jax.Array, graphs.receivers)

    # Number of nodes/edges in the first graph of the sequence
    n_node_first = graphs.n_node[0]
    n_edge_first = graphs.n_edge[0]

    # 1. Roll the per-graph feature arrays (n_node, n_edge).
    rolled_n_node = jnp.roll(graphs.n_node, -1, axis=0)
    rolled_n_edge = jnp.roll(graphs.n_edge, -1, axis=0)

    # Set the last graph to be a single, masked-out node (padding).
    rolled_n_node = rolled_n_node.at[-1].set(1)
    rolled_n_edge = rolled_n_edge.at[-1].set(0)

    # 2. Roll the flattened node and edge arrays by the size of the first graph.
    rolled_nodes = jax.tree.map(lambda x: jnp.roll(x, -n_node_first, axis=0), nodes)
    rolled_edges = jax.tree.map(lambda x: jnp.roll(x, -n_edge_first, axis=0), edges)
    rolled_senders = jnp.roll(senders, -n_edge_first, axis=0)
    rolled_receivers = jnp.roll(receivers, -n_edge_first, axis=0)

    # 3. Pad the last step of the sequence.
    # The last graph now occupies the slot of the rolled first graph.
    # We create a boolean mask to identify the elements that need to be padded.
    total_nodes = graphs.nodes["type_idx"].shape[0]  # type: ignore[operator]
    total_edges = graphs.senders.shape[0]  # type: ignore[operator]

    # Create a mask for the padding area
    node_indices = jnp.arange(total_nodes)
    edge_indices = jnp.arange(total_edges)
    node_pad_mask = node_indices >= (total_nodes - n_node_first)
    edge_pad_mask = edge_indices >= (total_edges - n_edge_first)

    # Use jnp.where to apply padding. This is JIT-compatible.
    padded_nodes = jax.tree.map(
        lambda x: jnp.where(node_pad_mask, -1, x),
        {k: v for k, v in rolled_nodes.items() if k != "mask"},
    )
    padded_nodes["mask"] = jnp.where(node_pad_mask, False, rolled_nodes["mask"])

    padded_edges = jax.tree.map(
        lambda x: jnp.where(edge_pad_mask, False, x), rolled_edges
    )

    # 4. Adjust sender/receiver indices for the non-padding edges that shifted forward.
    padded_senders = jnp.where(padded_edges["mask"], rolled_senders - n_node_first, 0)
    padded_receivers = jnp.where(
        padded_edges["mask"], rolled_receivers - n_node_first, 0
    )

    return graphs._replace(
        nodes=padded_nodes,
        edges=padded_edges,
        senders=padded_senders,
        receivers=padded_receivers,
        n_node=rolled_n_node,
        n_edge=rolled_n_edge,
    )


class JaxGraphReachAvoidSequence(JaxReachAvoidSequence):
    """Jax representation of a reach-avoid sequence with assignments and graphs."""

    # Graph-based representation
    reach_graphs: jraph.GraphsTuple
    avoid_graphs: jraph.GraphsTuple

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    @override
    def advance(self) -> "JaxGraphReachAvoidSequence":
        """Advance the reach-avoid sequence by one step. Returns a new sequence."""
        if self.reach_graphs.n_node.ndim > 1:
            raise ValueError(
                "The `advance` method is only supported for unbatched sequences "
                "(created with `from_seq`), not for batched sequences "
                "(created with `from_state_to_seqs`)."
            )

        is_last_step = self.depth == 1
        should_repeat = jnp.logical_and(
            is_last_step, self.last_index + 1 < self.repeat_last
        )

        def _repeat_step():
            return replace(self, last_index=self.last_index + 1)

        def _advance_step():
            # Advance assignment arrays one step
            new_reach = jnp.roll(self.reach, -1, axis=-2)
            new_avoid = jnp.roll(self.avoid, -1, axis=-2)

            # Pad the last row with -1s
            new_reach = new_reach.at[-1, :].set(-1)
            new_avoid = new_avoid.at[-1, :].set(-1)

            # Advance graph arrays one step
            new_reach_graphs = _roll_graphs(self.reach_graphs)
            new_avoid_graphs = _roll_graphs(self.avoid_graphs)

            return JaxGraphReachAvoidSequence(
                reach=new_reach,
                avoid=new_avoid,
                reach_graphs=new_reach_graphs,
                avoid_graphs=new_avoid_graphs,
                repeat_last=self.repeat_last,
                last_index=jnp.zeros_like(self.last_index),
            )

        return jax.lax.cond(
            jnp.all(should_repeat),
            _repeat_step,
            _advance_step,
        )

    @classmethod
    def from_seq(
        cls,
        seq: GraphReachAvoidSequence,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_nodes: int,
        max_edges: int,
    ) -> "JaxGraphReachAvoidSequence":
        """
        Converts a single GraphReachAvoidSequence into a batched Jax representation.
        """
        seq_len = len(seq.reach_avoid)

        # --- Pre-computation for efficiency ---
        assignment_map = {name: i for i, name in enumerate(assignments)}
        epsilon_idx = len(assignments)

        # --- Assignment processing ---
        reach_assign = -np.ones((seq_len, len(assignments)), dtype=np.int32)
        avoid_assign = -np.ones_like(reach_assign)

        for t_idx, (r, a) in enumerate(seq.reach_avoid):
            if isinstance(r, EpsilonType):
                reach_assign[t_idx, 0] = epsilon_idx
            else:
                for j, assign in enumerate(r):
                    reach_assign[t_idx, j] = assignment_map[assign]
            for j, assign in enumerate(a):
                avoid_assign[t_idx, j] = assignment_map[assign]

        # --- Graph processing (Ragged Strategy) ---
        # Initialize lists to hold data for each graph in the sequence
        reach_graph_parts = [
            _convert_to_arrays(g, propositions) for g, _ in seq.reach_avoid_graphs
        ]
        avoid_graph_parts = [
            _convert_to_arrays(g, propositions) for _, g in seq.reach_avoid_graphs
        ]

        # Concatenate parts to form the full graph tuple data
        reach_graphs = _build_graph_tuple_from_parts(
            reach_graph_parts, max_nodes, max_edges
        )
        avoid_graphs = _build_graph_tuple_from_parts(
            avoid_graph_parts, max_nodes, max_edges
        )

        return cls(
            reach=jnp.array(reach_assign),
            avoid=jnp.array(avoid_assign),
            reach_graphs=reach_graphs,
            avoid_graphs=avoid_graphs,
            repeat_last=jnp.array(seq.repeat_last, dtype=jnp.int32),
            last_index=jnp.array(0, dtype=jnp.int32),
        )

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[GraphReachAvoidSequence]],
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        max_nodes: int,
        max_edges: int,
    ) -> "JaxGraphReachAvoidSequence":
        """
        Converts a mapping from LDBA states to lists of GraphReachAvoidSequences
        into a batched Jax representation.
        """
        num_states = len(state_to_seqs)
        max_seqs = max((len(seqs) for seqs in state_to_seqs.values()), default=0)
        max_len = max(
            (len(s.reach_avoid) for seqs in state_to_seqs.values() for s in seqs),
            default=0,
        )

        # --- Pre-computation for efficiency ---
        assignment_map = {name: i for i, name in enumerate(assignments)}
        epsilon_idx = len(assignments)

        # --- Assignment processing ---
        reach_assign = -np.ones(
            (num_states, max_seqs, max_len, len(assignments)), dtype=np.int32
        )
        avoid_assign = -np.ones_like(reach_assign)
        repeat_last_arr = np.ones((num_states, max_seqs), dtype=np.int32)

        for state, seqs in state_to_seqs.items():
            for s_idx, seq in enumerate(seqs):
                repeat_last_arr[state, s_idx] = seq.repeat_last
                for t_idx, (r, a) in enumerate(seq.reach_avoid):
                    if isinstance(r, EpsilonType):
                        reach_assign[state, s_idx, t_idx, 0] = epsilon_idx
                    else:
                        for j, assign in enumerate(r):
                            reach_assign[state, s_idx, t_idx, j] = assignment_map[
                                assign
                            ]
                    for j, assign in enumerate(a):
                        avoid_assign[state, s_idx, t_idx, j] = assignment_map[assign]

        # --- Graph processing (Ragged Strategy) ---
        # Initialize final padded arrays and n_node/n_edge arrays
        all_reach_nodes = {
            "type_idx": -np.ones((num_states, max_seqs, max_nodes), dtype=np.int32),
            "prop_idx": -np.ones((num_states, max_seqs, max_nodes), dtype=np.int32),
            "mask": np.zeros((num_states, max_seqs, max_nodes), dtype=np.bool_),
        }
        all_reach_edges = {
            "mask": np.zeros((num_states, max_seqs, max_edges), dtype=np.bool_)
        }
        all_reach_senders = np.zeros((num_states, max_seqs, max_edges), dtype=np.int32)
        all_reach_receivers = np.zeros(
            (num_states, max_seqs, max_edges), dtype=np.int32
        )
        all_reach_n_node = np.zeros((num_states, max_seqs, max_len), dtype=np.int32)
        all_reach_n_edge = np.zeros((num_states, max_seqs, max_len), dtype=np.int32)

        all_avoid_nodes = jax.tree.map(np.copy, all_reach_nodes)
        all_avoid_edges = jax.tree.map(np.copy, all_reach_edges)
        all_avoid_senders = np.copy(all_reach_senders)
        all_avoid_receivers = np.copy(all_reach_receivers)
        all_avoid_n_node = np.copy(all_reach_n_node)
        all_avoid_n_edge = np.copy(all_reach_n_edge)

        # This loop is slow but runs only once at initialization.
        for state in range(num_states):
            for s_idx in range(max_seqs):
                try:
                    seq = state_to_seqs[state][s_idx]
                except (KeyError, IndexError):
                    seq = None

                # --- Process Reach Graphs for the sequence ---
                r_parts = [
                    _convert_to_arrays(
                        seq.reach_avoid_graphs[t][0] if seq and t < len(seq) else None,
                        propositions,
                    )
                    for t in range(max_len)
                ]
                _fill_batched_graph_parts(
                    r_parts,
                    state,
                    s_idx,
                    all_reach_nodes,
                    all_reach_edges,
                    all_reach_senders,
                    all_reach_receivers,
                    all_reach_n_node,
                    all_reach_n_edge,
                    max_nodes,
                    max_edges,
                )

                # --- Process Avoid Graphs for the sequence ---
                a_parts = [
                    _convert_to_arrays(
                        seq.reach_avoid_graphs[t][1] if seq and t < len(seq) else None,
                        propositions,
                    )
                    for t in range(max_len)
                ]
                _fill_batched_graph_parts(
                    a_parts,
                    state,
                    s_idx,
                    all_avoid_nodes,
                    all_avoid_edges,
                    all_avoid_senders,
                    all_avoid_receivers,
                    all_avoid_n_node,
                    all_avoid_n_edge,
                    max_nodes,
                    max_edges,
                )

        # Reshape n_node/n_edge to have batch dimensions
        reach_graphs = jraph.GraphsTuple(
            nodes=all_reach_nodes,
            edges=all_reach_edges,
            senders=all_reach_senders,  # type: ignore[operator]
            receivers=all_reach_receivers,  # type: ignore[operator]
            n_node=all_reach_n_node,  # type: ignore[operator]
            n_edge=all_reach_n_edge,  # type: ignore[operator]
            globals=None,
        )
        avoid_graphs = jraph.GraphsTuple(
            nodes=all_avoid_nodes,
            edges=all_avoid_edges,
            senders=all_avoid_senders,  # type: ignore[operator]
            receivers=all_avoid_receivers,  # type: ignore[operator]
            n_node=all_avoid_n_node,  # type: ignore[operator]
            n_edge=all_avoid_n_edge,  # type: ignore[operator]
            globals=None,
        )

        return cls(
            reach=jnp.array(reach_assign),
            avoid=jnp.array(avoid_assign),
            reach_graphs=reach_graphs,
            avoid_graphs=avoid_graphs,
            repeat_last=jnp.array(repeat_last_arr),
            last_index=jnp.zeros((num_states, max_seqs), dtype=jnp.int32),
        )


def _build_graph_tuple_from_parts(
    graph_parts: list, max_nodes: int, max_edges: int
) -> jraph.GraphsTuple:
    """Builds a single GraphsTuple from a list of graph parts."""
    nodes_list, edges_list, senders_list, receivers_list, n_node_list, n_edge_list = (
        zip(*graph_parts, strict=True)
    )

    # Concatenate all parts
    cat_nodes = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *nodes_list)
    cat_edges = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *edges_list)

    # Adjust sender/receiver indices
    node_offsets = np.concatenate([[0], np.cumsum(n_node_list)[:-1]])
    adj_senders = np.concatenate(
        [s + offset for s, offset in zip(senders_list, node_offsets, strict=True)]
    )
    adj_receivers = np.concatenate(
        [r + offset for r, offset in zip(receivers_list, node_offsets, strict=True)]
    )

    # Pad to max_nodes and max_edges
    num_nodes = cat_nodes["mask"].shape[0]
    num_edges = adj_senders.shape[0]

    if num_nodes > max_nodes or num_edges > max_edges:
        raise ValueError("Exceeded max_nodes or max_edges for the sequence.")

    # Pad nodes
    node_pad_len = max_nodes - num_nodes
    padded_nodes = jax.tree.map(
        lambda x: np.pad(x, ((0, node_pad_len), (0, 0)), constant_values=-1),
        {k: v for k, v in cat_nodes.items() if k != "mask"},
    )
    padded_nodes["mask"] = np.pad(
        cat_nodes["mask"], ((0, node_pad_len), (0, 0)), constant_values=False
    )

    # Pad edges
    edge_pad_len = max_edges - num_edges
    padded_edges = jax.tree.map(
        lambda x: np.pad(x, ((0, edge_pad_len), (0, 0)), constant_values=False),
        cat_edges,
    )
    padded_senders = np.pad(adj_senders, (0, edge_pad_len), constant_values=0)
    padded_receivers = np.pad(adj_receivers, (0, edge_pad_len), constant_values=0)

    return jraph.GraphsTuple(
        nodes=jax.tree.map(lambda x: x.reshape(-1), padded_nodes),
        edges=jax.tree.map(lambda x: x.reshape(-1), padded_edges),
        senders=padded_senders,  # type: ignore[operator]
        receivers=padded_receivers,  # type: ignore[operator]
        n_node=np.array(n_node_list, dtype=np.int32),  # type: ignore[operator]
        n_edge=np.array(n_edge_list, dtype=np.int32),  # type: ignore[operator]
        globals=None,
    )


def _fill_batched_graph_parts(
    graph_parts: list,
    state_idx: int,
    seq_idx: int,
    all_nodes: dict,
    all_edges: dict,
    all_senders: np.ndarray,
    all_receivers: np.ndarray,
    all_n_node: np.ndarray,
    all_n_edge: np.ndarray,
    max_nodes: int,
    max_edges: int,
):
    """Helper to fill pre-allocated arrays for one sequence."""
    (
        nodes_list,
        edges_list,
        senders_list,
        receivers_list,
        n_node_list,
        n_edge_list,
    ) = zip(*graph_parts, strict=True)

    # Concatenate parts for this sequence
    cat_nodes = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *nodes_list)
    cat_edges = jax.tree.map(lambda *x: np.concatenate(x, axis=0), *edges_list)
    node_offsets = np.concatenate([[0], np.cumsum(n_node_list)[:-1]])
    adj_senders = np.concatenate(
        [s + offset for s, offset in zip(senders_list, node_offsets, strict=True)]
    )
    adj_receivers = np.concatenate(
        [r + offset for r, offset in zip(receivers_list, node_offsets, strict=True)]
    )

    num_nodes = cat_nodes["mask"].shape[0]
    num_edges = adj_senders.shape[0]

    if num_nodes > max_nodes or num_edges > max_edges:
        raise ValueError("Exceeded max_nodes or max_edges for a sequence.")

    # Place into final batched arrays
    for key, arr in all_nodes.items():
        arr[state_idx, seq_idx, :num_nodes] = cat_nodes[key].flatten()
    for key, arr in all_edges.items():
        arr[state_idx, seq_idx, :num_edges] = cat_edges[key].flatten()

    all_senders[state_idx, seq_idx, :num_edges] = adj_senders
    all_receivers[state_idx, seq_idx, :num_edges] = adj_receivers
    all_n_node[state_idx, seq_idx, :] = n_node_list
    all_n_edge[state_idx, seq_idx, :] = n_edge_list


def _convert_to_arrays(
    graph_root: Node | EpsilonType | None,
    propositions: Sequence[str],
):
    """Converts a single boolean formula graph to padded numpy arrays."""
    if graph_root is None:  # Padding graph
        nodes = {
            "type_idx": np.array([[-1]], dtype=np.int32),
            "prop_idx": np.array([[-1]], dtype=np.int32),
            "mask": np.array([[False]], dtype=np.bool_),
        }
        edges = {"mask": np.empty((0, 1), dtype=np.bool_)}
        senders = np.array([], dtype=np.int32)
        receivers = np.array([], dtype=np.int32)
        n_node = np.array(1)
        n_edge = np.array(0)
        return nodes, edges, senders, receivers, n_node, n_edge

    node_map: dict[Node | EpsilonType, int] = {}
    node_features: list[list[int]] = []
    senders, receivers = [], []

    def add_node(node: Node | EpsilonType) -> int:
        if node in node_map:
            return node_map[node]
        idx = len(node_map)
        node_map[node] = idx
        node_features.append(_get_node_features_as_int(node, propositions))
        return idx

    def build_graph(node: Node | EpsilonType):
        parent_idx = add_node(node)
        children: Sequence[Node] = []
        if isinstance(node, MultiAndNode | MultiOrNode):
            children = node.operands
        elif isinstance(node, AndNode | OrNode):
            children = [node.left, node.right]
        elif isinstance(node, NotNode):
            children = [node.operand]

        for child in children:
            child_idx = build_graph(child)
            senders.append(child_idx)
            receivers.append(parent_idx)

        return parent_idx

    _ = build_graph(graph_root)

    num_nodes = len(node_features)
    num_edges = len(senders)

    # Create node feature arrays (unpadded)
    type_arr = -np.ones((num_nodes, 1), dtype=np.int32)
    prop_arr = -np.ones((num_nodes, 1), dtype=np.int32)
    node_mask_arr = np.zeros((num_nodes, 1), dtype=np.bool_)

    if num_nodes > 0:
        features_arr = np.array(node_features, dtype=np.int32)
        type_arr[:, 0] = features_arr[:, 0]
        prop_arr[:, 0] = features_arr[:, 1]
        node_mask_arr[:, 0] = True

    nodes = {"type_idx": type_arr, "prop_idx": prop_arr, "mask": node_mask_arr}
    edges = {"mask": np.ones((num_edges, 1), dtype=np.bool_)}

    # Create edge arrays (unpadded)
    senders_arr = np.array(senders, dtype=np.int32)
    receivers_arr = np.array(receivers, dtype=np.int32)

    return (
        nodes,
        edges,
        senders_arr,
        receivers_arr,
        np.array(num_nodes),
        np.array(num_edges),
    )


def _get_node_features_as_int(
    node: Node | EpsilonType, propositions: Sequence[str]
) -> list[int]:
    """Creates an integer feature vector for a graph node.
    Returns:
        A list of two integers: [type_id, prop_id].
        type_id is -1 for non-special nodes.
        prop_id is -1 for non-proposition nodes.
    """
    prop_id = -1
    if isinstance(node, VarNode):
        type_id = -1
        try:
            prop_id = propositions.index(node.name)
        except ValueError as err:
            raise ValueError(
                f"Proposition '{node.name}' not in environment propositions."
            ) from err
    elif isinstance(node, AndNode | MultiAndNode):
        type_id = NODE_TYPE_AND
    elif isinstance(node, OrNode | MultiOrNode):
        type_id = NODE_TYPE_OR
    elif isinstance(node, NotNode):
        type_id = NODE_TYPE_NOT
    elif isinstance(node, EmptyNode):
        type_id = NODE_TYPE_EMPTY
    elif isinstance(node, EpsilonType):
        type_id = NODE_TYPE_EPSILON
    else:
        raise TypeError(f"Unknown node type for featurization: {type(node)}")

    return [type_id, prop_id]
