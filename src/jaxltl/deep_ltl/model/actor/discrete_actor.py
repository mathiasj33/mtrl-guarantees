from collections.abc import Callable

import distrax
import equinox as eqx
import jax
from jax.nn.initializers import Initializer

from jaxltl.deep_ltl.model.epsilon_distribution import EpsilonDistribution
from jaxltl.networks.mlp import MLP


class DiscreteActor(eqx.Module):
    encoder: MLP
    action_probs: MLP
    epsilon_prob: MLP

    def __init__(
        self,
        in_size: int,
        num_actions: int,
        hidden_sizes: list[int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
    ):
        enc_key, action_key, eps_key = jax.random.split(key, 3)
        self.encoder = MLP(
            in_size,
            hidden_sizes[-1],
            hidden_sizes[:-1],
            activation,
            weight_init,
            bias_init,
            final_layer_activation=True,
            key=enc_key,
        )
        self.action_probs = MLP(
            hidden_sizes[-1],
            num_actions,
            [],
            weight_init=weight_init,
            bias_init=bias_init,
            final_layer_activation=False,
            key=action_key,
        )
        self.epsilon_prob = MLP(
            hidden_sizes[-1],
            1,
            [],
            weight_init=weight_init,
            bias_init=bias_init,
            final_layer_activation=False,
            key=eps_key,
        )

    def __call__(
        self, features: jax.Array, epsilon_mask: jax.Array
    ) -> distrax.Distribution:
        """Input shape: (batch_size, in_size).

        Input has to be batched because distrax distributions are not compatible with vmap.
        """
        encoded = jax.vmap(self.encoder)(features)
        action_probs = jax.vmap(self.action_probs)(encoded)
        log_eps = jax.vmap(self.epsilon_prob)(encoded)
        action_dist = distrax.Categorical(logits=action_probs)
        return EpsilonDistribution(action_dist, log_eps.squeeze(-1), epsilon_mask)
