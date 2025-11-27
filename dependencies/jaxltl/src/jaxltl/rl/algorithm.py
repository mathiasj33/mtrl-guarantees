from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import PyTree

from jaxltl.rl.actor_critic import ActorCritic


class RLAlgorithm(eqx.Module):
    """Base class for RL algorithms."""

    @abstractmethod
    def train(
        self,
        model: ActorCritic,
        env,
        env_params: PyTree,
        key: jax.Array,
        callback: Callable | None = None,
        callback_freq: int | None = None,
        seed: jax.Array | None = None,
    ) -> tuple[ActorCritic, dict]:
        """Train the model. Jittable.

        Args:
            model: The actor-critic model to be trained.
            env: The environment to train on.
            env_params: The parameters of the environment.
            key: A JAX random key.
            callback: An optional callback function to be called every `callback_freq` steps.
            callback_freq: The frequency (in interaction steps) at which to call the callback.
            seed: The seed associated with this run, can be used in the callback.

        Returns:
            A tuple containing the trained model and a dictionary of training metrics."""
        pass
