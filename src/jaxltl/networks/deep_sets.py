from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from jaxltl.networks.callable_module import CallableModule
from jaxltl.networks.mlp import MLP


class DeepSets(CallableModule):
    """Simple Deep Sets implementation. Learns a separate embedding for the empty set."""

    empty_embedding: jax.Array  # (embedding_dim,)
    mlp: MLP

    def __init__(
        self,
        embedding_dim: int,
        out_size: int,
        hidden_sizes: list[int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        weight_init: Initializer | None = None,
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        empty_key, mlp_key = jax.random.split(key)
        self.empty_embedding = jax.random.normal(empty_key, (embedding_dim,))
        self.mlp = MLP(
            embedding_dim,
            out_size,
            hidden_sizes,
            activation,
            weight_init,
            bias_init,
            final_layer_activation=True,
            key=mlp_key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Input shape: (max_set_size,embedding_dim) with 0s for padding."""
        emb = jax.lax.cond(
            jnp.all(x == 0),
            lambda: self.empty_embedding,
            lambda: jnp.sum(x, axis=0),
        )
        return self.mlp(emb)
