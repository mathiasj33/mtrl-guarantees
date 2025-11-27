"""Equinox-compatible lax utilities."""

from collections.abc import Callable

import equinox as eqx
import jax


def filter_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Y]:
    """A wrapper around jax.lax.scan that supports equinox modules."""
    carry_params, carry_static = eqx.partition(init, eqx.is_array)

    def aux(carry_params, x):
        carry = eqx.combine(carry_params, carry_static)
        carry, y = f(carry, x)
        carry_params, _ = eqx.partition(carry, eqx.is_array)
        return carry_params, y

    carry_params, y = jax.lax.scan(
        aux,
        carry_params,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )
    carry = eqx.combine(carry_params, carry_static)
    return carry, y
