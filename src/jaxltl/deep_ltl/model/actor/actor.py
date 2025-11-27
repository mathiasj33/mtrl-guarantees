from abc import abstractmethod

import distrax
import equinox as eqx
import jax


class Actor(eqx.Module):
    """Abstract base class for actors."""

    @abstractmethod
    def __call__(
        self, features: jax.Array, epsilon_mask: jax.Array
    ) -> distrax.Distribution:
        """Input shape: (batch_size, in_size).

        Input has to be batched because distrax distributions are not compatible with vmap.
        """
        pass
