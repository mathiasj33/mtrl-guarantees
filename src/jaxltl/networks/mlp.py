from collections.abc import Callable
from typing import override

import equinox as eqx
import jax
from jax.nn.initializers import Initializer

from jaxltl.networks.callable_module import CallableModule
from jaxltl.networks.network_utils import make_linear


class MLP(CallableModule):
    """Multi-layer perceptron (MLP) network."""

    layers: list[eqx.nn.Linear]
    activation: Callable[[jax.Array], jax.Array]
    final_layer_activation: bool
    output_size: int

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
        self.output_size = out_size

    @override
    @eqx.filter_jit
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.final_layer_activation:
                x = self.activation(x)
        return x
