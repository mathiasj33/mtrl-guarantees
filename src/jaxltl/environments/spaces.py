"""Spaces supported in jaxltl environments.

Adapted from gymnax (https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/spaces.py)."""

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


class Space(eqx.Module):
    """Minimal jittable class for abstract space."""

    shape: eqx.AbstractVar[tuple]

    @abstractmethod
    def sample(self, key: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def contains(self, x: jax.Array) -> Any:
        pass


class Discrete(Space):
    """Minimal jittable class for discrete spaces."""

    n: int
    shape: tuple = ()
    dtype = jnp.int32

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            key, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        x = x.astype(jnp.int32)
        return jnp.logical_and(x >= 0, x < self.n)


class Box(Space):
    """Minimal jittable class for array-shaped spaces."""

    low: jax.Array | float
    high: jax.Array | float
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            key, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        return jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
