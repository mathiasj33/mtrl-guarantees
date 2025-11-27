from abc import abstractmethod

import distrax
import equinox as eqx
import jax
from jaxtyping import PyTree


class ActorCritic(eqx.Module):
    def __call__(self, obs: PyTree) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass through the actor and critic networks.

        Args:
            obs: Batched observations.
            epsilon_mask: Batched mask whether epsilon actions are enabled.

        Returns:
            A tuple of (action distribution, state value).
        """
        features = self._compute_common_features(obs)
        dist = self._get_action(features, obs.epsilon_mask)
        value = self._get_value(features)
        return dist, value

    def get_action(self, obs: PyTree) -> distrax.Distribution:
        """Get action distribution from the actor network.

        Args:
            obs: Batched observations.
            epsilon_mask: Batched mask whether epsilon actions are enabled.

        Returns:
            Batched action distribution.
        """
        return self._get_action(self._compute_common_features(obs), obs.epsilon_mask)

    def get_value(self, obs: PyTree) -> jax.Array:
        """Get state value from the critic network.

        Args:
            obs: Batched observations.

        Returns:
            State values.
        """
        return self._get_value(self._compute_common_features(obs))

    @abstractmethod
    def _get_action(
        self, features: jax.Array, epsilon_mask: jax.Array
    ) -> distrax.Distribution:
        """Get action distribution from the actor network given features.

        Args:
            features: Batched features.
            epsilon_mask: Batched mask whether epsilon actions are enabled.

        Returns:
            Batched action distribution.
        """
        pass

    @abstractmethod
    def _get_value(self, features: jax.Array) -> jax.Array:
        """Get state value from the critic network.

        Args:
            features: Batched features.

        Returns:
            Batched state values.
        """
        pass

    @abstractmethod
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        """Compute common features from observations for actor and critic.

        Args:
            obs: Batched observations.

        Returns:
            Common features.
        """
        pass
