"""Utility functions for JAX-friendly sequence sampling."""

import jax
import jax.numpy as jnp


@jax.jit
def sample_assignments(
    available_mask: jax.Array, bounds: tuple[int, int], key: jax.Array
) -> jax.Array:
    """
    Samples between (min, max) assignments from the available ones.

    Args:
        available_mask: A boolean array of shape (num_assignments,) indicating which
            assignments are available for sampling.
        bounds: A tuple (min, max) specifying the range for
            the number of assignments to sample.
        key: A JAX random key.

    Returns:
        A boolean array of shape (num_assignments,) indicating which assignments were
        sampled.
    """
    nr_key, reach_key = jax.random.split(key)
    nr = jax.random.randint(nr_key, (), bounds[0], bounds[1] + 1)
    num_assignments = available_mask.shape[0]

    shuffled_indices = jax.random.permutation(reach_key, num_assignments)
    is_available = available_mask[shuffled_indices]
    cumulative_available = jnp.cumsum(is_available)
    is_in_sample_sorted = (cumulative_available <= nr) & is_available
    original_indices_order = jnp.argsort(shuffled_indices)
    mask = is_in_sample_sorted[original_indices_order]
    return mask


def sample_propositions(
    key: jax.Array, available_mask: jax.Array, bounds: tuple[int, int]
) -> jax.Array:
    """
    Samples between (min, max) propositions from the available ones.

    Args:
        key: A JAX random key.
        available_mask: A boolean array of shape (num_propositions,) indicating which
            propositions are available for sampling.
        bounds: A tuple (min, max) specifying the range for the number of
            propositions to sample.

    Returns:
        A boolean array of shape (num_propositions,) indicating which propositions
        were sampled.
    """
    nr_key, sample_key = jax.random.split(key)
    nr = jax.random.randint(nr_key, (), bounds[0], bounds[1] + 1)
    num_propositions = available_mask.shape[0]

    shuffled_indices = jax.random.permutation(sample_key, num_propositions)
    is_available = available_mask[shuffled_indices]
    cumulative_available = jnp.cumsum(is_available)
    is_in_sample_sorted = (cumulative_available <= nr) & is_available
    original_indices_order = jnp.argsort(shuffled_indices)
    mask = is_in_sample_sorted[original_indices_order]
    return mask
