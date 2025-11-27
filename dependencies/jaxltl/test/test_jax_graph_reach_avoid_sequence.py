import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.curriculum.zone_env_graph_samplers import (
    GraphZoneReachAvoidSampler,
    GraphZoneReachStaySampler,
)
from jaxltl.deep_ltl.eval.utils import _batch_graph_sequences
from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxltl.environments.zone_env.zone_env import ZoneEnv
from jaxltl.ltl.logic.assignment import Assignment

propositions = ZoneEnv.propositions
assignments = [Assignment(frozenset({color})) for color in propositions]
assignments.append(Assignment(frozenset()))  # empty assignment
_max_nodes = ZoneEnv.max_nodes
_max_edges = ZoneEnv.max_edges


# TODO: Add assertions


def test_sampler_to_jax_and_advance():
    sampler = GraphZoneReachAvoidSampler(
        depth=(1, 2),
        reach=(1, 2),
        avoid=(0, 2),
        propositions=propositions,
        assignments=assignments,
        max_length=3,
        max_nodes=_max_nodes,
        max_edges=_max_edges,
    )
    key = jax.random.key(0)

    for j in range(50):
        key, subkey = jax.random.split(key)
        jax_seq = sampler.sample(subkey)

        at_size_limit = (
            jnp.sum(jax_seq.reach_graphs.n_node) == _max_nodes
            or jnp.sum(jax_seq.avoid_graphs.n_node) == _max_nodes
            or jnp.sum(jax_seq.reach_graphs.n_edge) == _max_edges
            or jnp.sum(jax_seq.avoid_graphs.n_edge) == _max_edges
        )

        if j < 5 or at_size_limit:
            print(f"\n\n--- Reach-Avoid Sample {j + 1} ---")
            print("\n--- Initial JAX Assignments (Reach) ---")
            print(jax_seq.reach)
            print("\n--- Initial JAX Graph (Reach) ---")
            print(jax_seq.reach_graphs)
            print("\n--- Initial JAX Assignments (Avoid) ---")
            print(jax_seq.avoid)
            print("\n--- Initial JAX Graph (Avoid) ---")
            print(jax_seq.avoid_graphs)
            print("\n--- Initial Repeat last ---")
            print(jax_seq.repeat_last)
            print("\n--- Initial Last Index ---")
            print(jax_seq.last_index)

        # Advance the sequence
        advanced_seq = jax_seq.advance()

        if j < 5 or at_size_limit:
            print("\n--- Advanced JAX Assignments (Reach) ---")
            print(advanced_seq.reach)
            print("\n--- Advanced JAX Graph (Reach) ---")
            print(advanced_seq.reach_graphs)
            print("\n--- Advanced JAX Assignments (Avoid) ---")
            print(advanced_seq.avoid)
            print("\n--- Advanced JAX Graph (Avoid) ---")
            print(advanced_seq.avoid_graphs)
            print("\n--- Advanced Repeat last ---")
            print(advanced_seq.repeat_last)
            print("\n--- Advanced Last Index ---")
            print(advanced_seq.last_index)

        # Second advance
        advanced_seq = advanced_seq.advance()


def test_reach_stay_sampler_to_jax_and_advance():
    sampler = GraphZoneReachStaySampler(
        num_stay=60,
        avoid=(0, 2),
        propositions=propositions,
        assignments=assignments,
        max_length=3,
        max_nodes=_max_nodes,
        max_edges=_max_edges,
    )
    key = jax.random.key(42)

    for j in range(50):
        key, subkey = jax.random.split(key)
        jax_seq = sampler.sample(subkey)

        at_size_limit = (
            jnp.sum(jax_seq.reach_graphs.n_node) == _max_nodes
            or jnp.sum(jax_seq.avoid_graphs.n_node) == _max_nodes
            or jnp.sum(jax_seq.reach_graphs.n_edge) == _max_edges
            or jnp.sum(jax_seq.avoid_graphs.n_edge) == _max_edges
        )

        if j < 5 or at_size_limit:
            print(f"\n\n--- Reach-Avoid Sample {j + 1} ---")
            print("\n--- Initial JAX Assignments (Reach) ---")
            print(jax_seq.reach)
            print("\n--- Initial JAX Graph (Reach) ---")
            print(jax_seq.reach_graphs)
            print("\n--- Initial JAX Assignments (Avoid) ---")
            print(jax_seq.avoid)
            print("\n--- Initial JAX Graph (Avoid) ---")
            print(jax_seq.avoid_graphs)
            print("\n--- Initial Repeat last ---")
            print(jax_seq.repeat_last)
            print("\n--- Initial Last Index ---")
            print(jax_seq.last_index)

        # Advance the sequence
        advanced_seq = jax_seq.advance()

        if j < 5 or at_size_limit:
            print("\n--- Advanced JAX Assignments (Reach) ---")
            print(advanced_seq.reach)
            print("\n--- Advanced JAX Graph (Reach) ---")
            print(advanced_seq.reach_graphs)
            print("\n--- Advanced JAX Assignments (Avoid) ---")
            print(advanced_seq.avoid)
            print("\n--- Advanced JAX Graph (Avoid) ---")
            print(advanced_seq.avoid_graphs)
            print("\n--- Advanced Repeat last ---")
            print(advanced_seq.repeat_last)
            print("\n--- Advanced Last Index ---")
            print(advanced_seq.last_index)

        # Second advance
        advanced_seq = advanced_seq.advance()


def test_batch_to_jax():
    sampler = GraphZoneReachAvoidSampler(
        depth=(2, 2),
        reach=(1, 2),
        avoid=(0, 2),
        propositions=propositions,
        assignments=assignments,
        max_length=3,
        max_nodes=_max_nodes,
        max_edges=_max_edges,
    )
    key = jax.random.key(0)

    graph_seqs = []
    for _ in range(6):
        key, subkey = jax.random.split(key)
        graph_seq = sampler.sample_graph(subkey)
        graph_seqs.append(graph_seq)

    # Convert to JAX version
    state_to_seqs = {
        0: [graph_seqs[0], graph_seqs[1], graph_seqs[2]],
        1: [graph_seqs[3], graph_seqs[4], graph_seqs[5]],
    }
    jax_seq = JaxGraphReachAvoidSequence.from_state_to_seqs(
        state_to_seqs, propositions, assignments, _max_nodes, _max_edges
    )

    print("\n--- Initial JAX Assignments (Reach) ---")
    print(jax_seq.reach)
    print("\n--- Initial JAX Graph (Reach) ---")
    print(jax_seq.reach_graphs)
    print("\n--- Initial JAX Assignments (Avoid) ---")
    print(jax_seq.avoid)
    print("\n--- Initial JAX Graph (Avoid) ---")
    print(jax_seq.avoid_graphs)
    print("\n--- Initial Repeat last ---")
    print(jax_seq.repeat_last)
    print("\n--- Initial Last Index ---")
    print(jax_seq.last_index)

    graph_0_0 = jax.tree.map(lambda x: x[0, 0], jax_seq.reach_graphs)
    print("\n--- Sliced Graph (state=0, seq=0) ---")
    print(graph_0_0)


def test_batch_graph_sequences():
    sampler = GraphZoneReachAvoidSampler(
        depth=(2, 2),
        reach=(1, 2),
        avoid=(0, 2),
        propositions=propositions,
        assignments=assignments,
        max_length=3,
        max_nodes=_max_nodes,
        max_edges=_max_edges,
    )
    key = jax.random.key(0)

    jax_seqs = []
    for _ in range(3):
        graph_seqs = []
        for _ in range(6):
            key, subkey = jax.random.split(key)
            graph_seq = sampler.sample_graph(subkey)
            graph_seq.repeat_last = 1
            graph_seqs.append(graph_seq)

        # Convert to JAX version
        state_to_seqs = {
            0: [graph_seqs[0], graph_seqs[1], graph_seqs[2]],
            1: [graph_seqs[3], graph_seqs[4], graph_seqs[5]],
        }
        jax_seqs.append(
            JaxGraphReachAvoidSequence.from_state_to_seqs(
                state_to_seqs, propositions, assignments, _max_nodes, _max_edges
            )
        )

    batched_jax_seq = _batch_graph_sequences(jax_seqs)

    print("\n--- Initial JAX Assignments (Reach) ---")
    print(batched_jax_seq.reach)
    print("\n--- Initial JAX Graph (Reach) ---")
    print(batched_jax_seq.reach_graphs)
    print("\n--- Initial JAX Assignments (Avoid) ---")
    print(batched_jax_seq.avoid)
    print("\n--- Initial JAX Graph (Avoid) ---")
    print(batched_jax_seq.avoid_graphs)
    print("\n--- Initial Repeat last ---")
    print(batched_jax_seq.repeat_last)
    print("\n--- Initial Last Index ---")
    print(batched_jax_seq.last_index)

    graph_0_0_0 = jax.tree.map(lambda x: x[0, 0, 0], batched_jax_seq.reach_graphs)
    print("\n--- Sliced Graph (batch=0, state=0, seq=0) ---")
    print(graph_0_0_0)
