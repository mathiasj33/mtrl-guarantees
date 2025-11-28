import jax
from distrax import Categorical, Distribution, Normal
from jax import numpy as jnp


class GaussianMixture(Distribution):
    """A simple mixture of Gaussian distributions.

    In contrast to distrax.MixtureSameFamily, this class supports cdf computation.
    """

    def __init__(self, weights: list[float], means: list[float], stddevs: list[float]):
        self.mixture = Categorical(probs=jnp.array(weights))
        self.components = Normal(
            loc=jnp.array(means).reshape(-1, 1), scale=jnp.array(stddevs).reshape(-1, 1)
        )

    def _sample_n(self, key, n):
        index_key, sample_key = jax.random.split(key)
        indices = self.mixture.sample(seed=index_key, sample_shape=(n,))
        samples = self.components.sample(seed=sample_key, sample_shape=(n,))
        return samples[jnp.arange(n), indices]

    def log_prob(self, value):
        return jax.scipy.special.logsumexp(
            a=self.components.log_prob(value),
            b=self.mixture.probs[:, None],  # type: ignore
            axis=0,
        )

    def log_cdf(self, value):
        return jax.scipy.special.logsumexp(
            a=self.components.log_cdf(value),
            b=self.mixture.probs[:, None],  # type: ignore
            axis=0,
        )

    def event_shape(self):
        return ()
