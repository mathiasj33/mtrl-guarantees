"""Utility functions for creating neural networks."""

import equinox as eqx
import jax
from jax.nn.initializers import Initializer


def make_linear(
    in_size: int,
    out_size: int,
    weight_init: Initializer | None,
    bias_init: Initializer | None,
    *,
    key: jax.Array,
) -> eqx.nn.Linear:
    key1, key2, key3 = jax.random.split(key, 3)
    layer = eqx.nn.Linear(in_size, out_size, key=key1)
    if weight_init is not None:
        new_weight = weight_init(key2, layer.weight.shape)
        layer = eqx.tree_at(lambda layer: layer.weight, layer, new_weight)
    if bias_init is not None and layer.bias is not None:
        new_bias = bias_init(key3, layer.bias.shape)
        layer = eqx.tree_at(lambda layer: layer.bias, layer, new_bias)
    return layer
