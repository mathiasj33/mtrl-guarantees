import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def add_batch_dim(module: eqx.Module, batch_size: int) -> eqx.Module:
    """Add a batch dimension to all array fields of an Equinox Module.

    Args:
        module: The Equinox Module to add a batch dimension to.
        batch_size: The size of the batch dimension to add.

    Returns:
        A new Equinox Module with a batch dimension added to all array fields.
    """
    params, static = eqx.partition(module, eqx.is_array)
    batched_params = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None, ...], (batch_size,) + x.shape), params
    )
    return eqx.combine(batched_params, static)


def pytree_where(condition: jax.Array, x: PyTree, y: PyTree) -> PyTree:
    """Generalization of jnp.where to pytrees.

    Args:
        condition: A boolean array indicating which elements to select from `x` and `y`.
        x: The first pytree to select elements from.
        y: The second pytree to select elements from.

        condition needs to be broadcastable to the shape of the leaves in x and y.

    Returns:
        A new pytree with elements selected from `x` and `y` based on `condition`.
    """
    return jax.tree.map(
        lambda a, b: jnp.where(condition.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
        x,
        y,
    )
