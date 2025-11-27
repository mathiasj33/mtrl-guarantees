from collections.abc import Callable
from typing import TypedDict, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from jax.nn.initializers import Initializer

from jaxltl.networks.callable_module import CallableModule
from jaxltl.networks.network_utils import make_linear


class NodeFeatures(TypedDict):
    features: jax.Array
    mask: jax.Array


class EdgeFeatures(TypedDict):
    mask: jax.Array


class GCN(CallableModule):
    """A simple Graph Convolutional Network."""

    layers: list[eqx.nn.Linear]
    activation: Callable[[jax.Array], jax.Array]
    final_layer_activation: bool

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: list[int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        final_layer_activation: bool = True,
        key: jax.Array,
    ):
        sizes = [in_size] + hidden_sizes + [out_size]
        linear_keys = jax.random.split(key, len(sizes) - 1)

        self.layers = [
            make_linear(
                sizes[i],
                sizes[i + 1],
                weight_init,
                bias_init,
                key=linear_keys[i],
            )
            for i in range(len(sizes) - 1)
        ]
        self.activation = activation
        self.final_layer_activation = final_layer_activation

    def _gcn_conv(
        self, layer: eqx.nn.Linear, graph: jraph.GraphsTuple
    ) -> jraph.GraphsTuple:
        """A single GCN layer operation with Symmetric Normalization."""
        nodes = cast(NodeFeatures, graph.nodes)
        edges = cast(EdgeFeatures, graph.edges)
        receivers = cast(jax.Array, graph.receivers)
        senders = cast(jax.Array, graph.senders)

        # 1. Setup Masks and Shapes
        node_mask = nodes["mask"]
        edge_mask = edges["mask"]
        total_num_nodes = nodes["features"].shape[0]

        # 2. Linear Transformation (W * H)
        # We mask input features to ensure clean padding
        features = jnp.where(node_mask[:, None], nodes["features"], 0.0)
        # We manually apply the weight matrix, ignoring the bias for now.
        # eqx.nn.Linear stores weights as (out, in), so we transpose.
        # (N, in) @ (in, out) -> (N, out)
        transformed_features = features @ layer.weight.T

        # 3. Calculate Symmetric Normalization Coeffs: 1 / sqrt(Degree)
        # GCN Formula: D^(-1/2) * A_hat * D^(-1/2)

        # A. Calculate degree of actual edges
        edge_weights = jnp.where(edge_mask, 1.0, 0.0)
        node_degree = jraph.segment_sum(
            edge_weights, receivers, num_segments=total_num_nodes
        )

        # B. Add 1.0 for the implicit self-loop
        total_degree = node_degree + 1.0

        # C. Calculate 1/sqrt(degree)
        # We add 1e-10 inside sqrt just to be safe against division by zero
        # (though total_degree should be >= 1.0)
        norm = jax.lax.rsqrt(total_degree)

        # Mask the norm (padding nodes should have 0 norm)
        norm = jnp.where(node_mask, norm, 0.0)

        # 4. Message Passing

        # A. Pre-normalize source features: H' = H * D^(-1/2)
        features_norm = transformed_features * norm[:, None]

        # B. Gather messages from neighbors
        messages = features_norm[senders]

        # C. Mask invalid edges
        messages = jnp.where(edge_mask[:, None], messages, 0.0)

        # D. Aggregate (Sum)
        aggr_part = jraph.segment_sum(messages, receivers, num_segments=total_num_nodes)

        # E. Add Self-Loop Contribution
        # Instead of adding a physical edge, we add the node's own pre-normalized features.
        # Total Sum = Sum(Neighbors) + Self
        total_aggr = aggr_part + features_norm

        # F. Post-normalize: H_new = Total_Sum * D^(-1/2)
        new_features = total_aggr * norm[:, None]

        # 5. Apply Bias (AFTER Aggregation)
        if layer.bias is not None:
            # Broadcast bias across nodes
            new_features = new_features + layer.bias

        # 6. Update Graph
        updated_nodes = nodes.copy()
        updated_nodes["features"] = new_features
        return graph._replace(nodes=updated_nodes)

    @override
    @eqx.filter_jit
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Processes the graph through all GCN layers."""

        for i, layer in enumerate(self.layers):
            graph = self._gcn_conv(layer, graph)

            if i < len(self.layers) - 1 or self.final_layer_activation:
                nodes = cast(NodeFeatures, graph.nodes)
                features = self.activation(nodes["features"])

                # Re-mask output to keep padding clean
                features = jnp.where(nodes["mask"][:, None], features, 0.0)

                new_nodes = nodes.copy()
                new_nodes["features"] = features
                graph = graph._replace(nodes=new_nodes)

        return graph
