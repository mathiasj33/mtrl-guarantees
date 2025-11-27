import jax
import jax.numpy as jnp
from distrax import Bernoulli, Distribution


class EpsilonDistribution(Distribution):
    """Models a policy that first decides whether to execute an epsilon action, and then
    samples from a different action distribution if not.

    Note: this is not a valid distribution in the mathematical sense, since the sample
    function has to return a tuple of (action, took_epsilon). The calling code must
    handle this appropriately, and discard the action when took_epsilon is True.
    """

    def __init__(
        self,
        action_dist: Distribution,
        epsilon_log_prob: jax.Array,
        epsilon_mask: jax.Array,
    ):
        """
        Args:
            action_dist: The distribution to sample from when not taking an epsilon action.
            epsilon_log_prob: The log probability of taking an epsilon action.
            epsilon_mask: A boolean mask indicating if epsilon actions are permissible.
        """
        self.action_dist = action_dist
        logits = jnp.where(epsilon_mask, epsilon_log_prob, -jnp.inf)
        self.epsilon_dist = Bernoulli(logits=logits)

    def _sample_n(self, key: jax.Array, n: int):
        action_key, eps_key = jax.random.split(key)
        actions = self.action_dist._sample_n(action_key, n)
        take_epsilon = self.epsilon_dist._sample_n(eps_key, n)
        return actions, take_epsilon

    def log_prob(self, value: tuple[jax.Array, jax.Array]):
        actions, take_epsilon = value
        return self.epsilon_dist.log_prob(take_epsilon) + (
            1 - take_epsilon
        ) * self.action_dist.log_prob(actions)

    def event_shape(self):
        return self.action_dist.event_shape(), ()

    def _sample_n_and_log_prob(self, key, n):
        action_key, eps_key = jax.random.split(key)
        env_samples, env_log_prob = self.action_dist._sample_n_and_log_prob(
            action_key, n
        )
        eps_samples, eps_log_prob = self.epsilon_dist._sample_n_and_log_prob(eps_key, n)
        log_prob = eps_log_prob + (1 - eps_samples) * env_log_prob
        return (env_samples, eps_samples), log_prob

    def entropy(self):
        eps_entropy = self.epsilon_dist.entropy()
        not_eps_prob = self.epsilon_dist.prob(0)
        return eps_entropy + not_eps_prob * self.action_dist.entropy()

    def mode(self):
        return self.action_dist.mode(), self.epsilon_dist.mode()
