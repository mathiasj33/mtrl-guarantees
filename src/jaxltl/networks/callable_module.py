from abc import abstractmethod

import equinox as eqx
import jax


class CallableModule(eqx.Module):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> jax.Array:
        """Perform a forward pass."""
        pass
