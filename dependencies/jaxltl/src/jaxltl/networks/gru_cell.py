import math
from typing import cast

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module
from equinox.nn._misc import default_init


class GRUCell(Module):
    """A single step of a Gated Recurrent Unit (GRU).

    Adapted from Equinox's implementation to include a bias for every hidden gate. This
    is closer to the original GRU formulation, and follows the PyTorch implementation.
    """

    weight_ih: jax.Array
    weight_hh: jax.Array
    bias: jax.Array | None
    bias_h: jax.Array | None
    input_size: int
    hidden_size: int
    use_bias: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        *,
        key: jax.Array,
    ):
        dtype = jnp.float32
        ihkey, hhkey, bkey, bkey2 = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        ihshape = (3 * hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (3 * hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)
        if use_bias:
            self.bias = default_init(bkey, (3 * hidden_size,), dtype, lim)
            self.bias_h = default_init(bkey2, (3 * hidden_size,), dtype, lim)
        else:
            self.bias = None
            self.bias_h = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def __call__(self, input: jax.Array, hidden: jax.Array) -> jax.Array:
        """Perform a single step of the GRU update.

        Args:
            input: shape (input_size,)
            hidden: shape (hidden_size,)

        Returns:
            new hidden state: shape (hidden_size,)
        """
        if self.use_bias:
            bias = cast(jax.Array, self.bias)
            bias_h = cast(jax.Array, self.bias_h)
        else:
            bias = 0
            bias_h = 0
        igates = jnp.split(self.weight_ih @ input + bias, 3)
        hgates = jnp.split(self.weight_hh @ hidden + bias_h, 3)
        reset = jnn.sigmoid(igates[0] + hgates[0])
        inp = jnn.sigmoid(igates[1] + hgates[1])
        new = jnn.tanh(igates[2] + reset * hgates[2])
        return new + inp * (hidden - new)
