import jax.numpy as jnp
import jraph


def test_graph_construction():
    # Define a three node graph, each node has an integer as its feature.
    node_features = jnp.array([[0.0], [1.0], [2.0]])

    # We will construct a graph fro which there is a directed edge between each node
    # and its successor. We define this with `senders` (source nodes) and `receivers`
    # (destination nodes).
    senders = jnp.array([0, 1, 2])
    receivers = jnp.array([1, 2, 0])

    # You can optionally add edge attributes.
    edges = jnp.array([[5.0], [6.0], [7.0]])

    # We then save the number of nodes and the number of edges.
    # This information is used to make running GNNs over multiple graphs
    # in a GraphsTuple possible.
    n_node = jnp.array([3])
    n_edge = jnp.array([3])

    # Optionally you can add `global` information, such as a graph label.

    global_context = jnp.array([[1]])  # Same feature dimensions as nodes and edges.
    graph = jraph.GraphsTuple(
        nodes=node_features,
        senders=senders,
        receivers=receivers,
        edges=edges,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context,
    )

    print(graph)

    # Batching

    two_graph_graphstuple = jraph.batch([graph, graph])

    print(two_graph_graphstuple)

    # Different sets of features as a PyTree

    node_targets = jnp.array([[True], [False], [True]])
    graph = graph._replace(nodes={"inputs": graph.nodes, "targets": node_targets})

    print(graph)
