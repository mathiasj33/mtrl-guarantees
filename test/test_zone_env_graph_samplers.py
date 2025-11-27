import jax

from jaxltl.deep_ltl.curriculum.zone_env_graph_samplers import (
    GraphZoneReachAvoidSampler,
    GraphZoneReachStaySampler,
)
from jaxltl.deep_ltl.reach_avoid.graph_reach_avoid_sequence import (
    GraphReachAvoidSequence,
)
from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import EPSILON, EpsilonType
from jaxltl.environments.zone_env.zone_env import ZoneEnv
from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import (
    AndNode,
    MultiAndNode,
    MultiOrNode,
    NotNode,
    OrNode,
    VarNode,
)

propositions = ZoneEnv.propositions
assignments = [Assignment(frozenset({color})) for color in propositions]
assignments.append(Assignment(frozenset()))  # empty assignment
_max_nodes = ZoneEnv.max_nodes
_max_edges = ZoneEnv.max_edges


def get_props_from_graph(graph):
    """Helper to recursively extract all proposition names from a graph."""
    if graph is None:
        return set()
    if isinstance(graph, VarNode):
        return {graph.name}
    if isinstance(graph, MultiAndNode | MultiOrNode):
        props = set()
        for op in graph.operands:
            props.update(get_props_from_graph(op))
        return props
    if isinstance(graph, AndNode | OrNode):
        return get_props_from_graph(graph.left) | get_props_from_graph(graph.right)
    if isinstance(graph, NotNode):
        return get_props_from_graph(graph.operand)
    return set()


def test_graph_reach_avoid_sampler():
    env = ZoneEnv()  # for reconstruction
    sampler = GraphZoneReachAvoidSampler(
        depth=(1, 3),
        reach=(1, 3),
        avoid=(0, 3),
        propositions=propositions,
        assignments=assignments,
        max_length=3,
        max_nodes=_max_nodes,
        max_edges=_max_edges,
    )
    key = jax.random.key(0)

    for j in range(50):
        key, subkey = jax.random.split(key)
        seq = sampler.sample_graph(subkey)
        reconstructed_seq = GraphReachAvoidSequence.from_reach_avoid_sequence(seq, env)

        if j < 5:
            print(f"\n--- Reach-Avoid Sample {j + 1} ---")
            print(f"Assignments: {seq}")
            print("Graphs:")
            for i, (rg, ag) in enumerate(seq.reach_avoid_graphs):
                print(f"  Step {i}: Reach={rg}, Avoid={ag}")
            print("Graphs Reconstructed from Assignment Sets:")
            for i, (rg, ag) in enumerate(reconstructed_seq.reach_avoid_graphs):
                print(f"  Step {i}: Reach={rg}, Avoid={ag}")

        # Check sequence length
        assert sampler.depth[0] <= len(seq.reach_avoid_graphs) <= sampler.depth[1]
        assert len(seq.reach_avoid_graphs) == len(seq.reach_avoid)

        assert len(seq.reach_avoid_graphs) == 3
        assert len(seq.reach_avoid) == 3

        used_props = set()
        for i in range(len(seq.reach_avoid_graphs)):
            reach_graph, avoid_graph = seq.reach_avoid_graphs[i]
            reach_assigns, avoid_assigns = seq.reach_avoid[i]

            # --- Graph Checks ---
            reach_props = get_props_from_graph(reach_graph)
            avoid_props = get_props_from_graph(avoid_graph)

            if reach_graph is not None:
                assert isinstance(reach_graph, VarNode | MultiOrNode)
                if isinstance(reach_graph, VarNode):
                    num_reach = 1
                else:
                    num_reach = len(reach_graph.operands)
                assert sampler.reach[0] <= num_reach <= sampler.reach[1]

            if avoid_graph is not None:
                assert isinstance(avoid_graph, VarNode | MultiOrNode)
                if isinstance(avoid_graph, VarNode):
                    num_avoid = 1
                else:
                    num_avoid = len(avoid_graph.operands)
                assert sampler.avoid[0] <= num_avoid <= sampler.avoid[1]

            # Check disjointness of propositions
            assert reach_props.isdisjoint(avoid_props)
            assert reach_props.isdisjoint(used_props)
            used_props.update(reach_props)
            used_props.update(avoid_props)

            # --- Consistency Checks ---
            if reach_graph is not None and not isinstance(reach_assigns, EpsilonType):
                for assignment in reach_assigns:
                    assert reach_graph.eval(assignment)
            if avoid_graph is not None:
                for assignment in avoid_assigns:
                    assert avoid_graph.eval(assignment)

            # Check that no other assignments satisfy the graphs
            if not isinstance(reach_assigns, EpsilonType):
                other_assignments = set(assignments) - reach_assigns
                for assignment in other_assignments:
                    if reach_graph is not None:
                        assert not reach_graph.eval(assignment)

            other_assignments = set(assignments) - avoid_assigns
            for assignment in other_assignments:
                if avoid_graph is not None:
                    assert not avoid_graph.eval(assignment)


def test_graph_reach_stay_sampler():
    env = ZoneEnv()  # for reconstruction
    sampler = GraphZoneReachStaySampler(
        num_stay=60,
        avoid=(0, 3),
        propositions=propositions,
        assignments=assignments,
        max_length=3,
        max_nodes=_max_nodes,
        max_edges=_max_edges,
    )
    key = jax.random.key(42)

    for j in range(50):
        key, subkey = jax.random.split(key)
        seq = sampler.sample_graph(subkey)
        reconstructed_seq = GraphReachAvoidSequence.from_reach_avoid_sequence(seq, env)

        if j < 5:
            print(f"\n--- Reach-Stay Sample {j + 1} ---")
            print(f"Assignments: {seq}")
            print("Graphs:")
            for i, (rg, ag) in enumerate(seq.reach_avoid_graphs):
                print(f"  Step {i}: Reach={rg}, Avoid={ag}")
            print("Graphs Reconstructed from Assignment Sets:")
            for i, (rg, ag) in enumerate(reconstructed_seq.reach_avoid_graphs):
                print(f"  Step {i}: Reach={rg}, Avoid={ag}")

        # Check sequence structure
        assert len(seq.reach_avoid_graphs) == 3
        assert len(seq.reach_avoid) == 3
        assert seq.repeat_last == sampler.num_stay

        # --- Check Step 0 (Initial Avoid) ---
        reach_graph_0, avoid_graph_0 = seq.reach_avoid_graphs[0]
        reach_assigns_0, avoid_assigns_0 = seq.reach_avoid[0]

        assert reach_graph_0 is EPSILON
        assert reach_assigns_0 is EPSILON

        if avoid_graph_0 is not None:
            assert isinstance(avoid_graph_0, VarNode | MultiOrNode)
            if isinstance(avoid_graph_0, VarNode):
                num_avoid = 1
            else:
                num_avoid = len(avoid_graph_0.operands)
            assert sampler.avoid[0] <= num_avoid <= sampler.avoid[1]

        # --- Check Step 1 (Reach and Stay) ---
        reach_graph_1, avoid_graph_1 = seq.reach_avoid_graphs[1]
        reach_assigns_1, avoid_assigns_1 = seq.reach_avoid[1]

        # Check reach graph is a single proposition
        assert isinstance(reach_graph_1, VarNode)
        reach_prop = reach_graph_1.name

        # Check avoid graph contains all other propositions
        assert isinstance(avoid_graph_1, NotNode)
        assert isinstance(avoid_graph_1.operand, VarNode)
        assert avoid_graph_1.operand.name == reach_prop

        # Check consistency
        if reach_graph_1 is not None and not isinstance(reach_assigns_1, EpsilonType):
            for assign in reach_assigns_1:
                assert reach_graph_1.eval(assign)
        if avoid_graph_1 is not None:
            for assign in avoid_assigns_1:
                assert avoid_graph_1.eval(assign)
