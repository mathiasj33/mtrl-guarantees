from collections.abc import Callable
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn

from jaxltl.networks.callable_module import CallableModule


class ConvNet(CallableModule):
    """Convolutional neural network."""

    layers: list[nn.Conv2d]
    activation: Callable[[jax.Array], jax.Array]
    final_layer_activation: bool
    output_size: int

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        channels: list[int],
        kernel_size: tuple[int, int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        *,
        final_layer_activation: bool = True,
        key: jax.Array,
    ):
        h, w, c = obs_shape
        channels = [c, *channels]
        layers = []
        for i in range(len(channels) - 1):
            key, subkey = jax.random.split(key)
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size, key=subkey)
            )
        self.layers = layers
        self.activation = activation
        self.final_layer_activation = final_layer_activation

        num_conv = len(channels) - 1
        k_h, k_w = kernel_size
        h_out = h - num_conv * (k_h - 1)
        w_out = w - num_conv * (k_w - 1)
        self.output_size = h_out * w_out * channels[-1]

    @override
    @eqx.filter_jit
    def __call__(self, x):
        x = jnp.transpose(x, (2, 0, 1))  # HWC to CHW
        x = x.astype(jnp.float32)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.final_layer_activation:
                x = self.activation(x)
        return x.flatten()
