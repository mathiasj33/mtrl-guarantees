from math import prod
from typing import NamedTuple, override

import distrax
import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxltl.deep_ltl.curriculum.sequence_sampler import JaxReachAvoidSequence
from jaxltl.deep_ltl.model.actor.actor import Actor
from jaxltl.environments import spaces
from jaxltl.environments.spaces import Space
from jaxltl.networks.conv_net import ConvNet
from jaxltl.networks.deep_sets import DeepSets
from jaxltl.networks.gru_cell import GRUCell
from jaxltl.networks.mlp import MLP
from jaxltl.rl.actor_critic import ActorCritic


class DeepLTLModel(ActorCritic):
    env_net: MLP | ConvNet
    embedding: nn.Embedding
    deep_sets: DeepSets
    gru: GRUCell
    actor: Actor
    critic: MLP

    _flatten_features: bool

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_space: Space,
        num_assignments: int,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        is_conv = "ConvNet" in config.env_net._target_
        params = {"obs_shape": obs_shape} if is_conv else {"in_size": prod(obs_shape)}
        self.env_net = hydra.utils.instantiate(config.env_net, **params, key=env_key)
        self._flatten_features = not is_conv
        key, emb_key = jax.random.split(key)
        embedding_dim = config.sequence.embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_assignments + 1,  # +1 for epsilon transitions
            embedding_size=embedding_dim,
            key=emb_key,
        )
        key, ds_key = jax.random.split(key)
        self.deep_sets = hydra.utils.instantiate(
            config.sequence.deep_sets,
            embedding_dim=embedding_dim,
            key=ds_key,
        )
        key, gru_key = jax.random.split(key)
        # the GRU takes both reach and avoid embeddings as input, hence 2 * embedding_dim
        self.gru = GRUCell(
            input_size=2 * config.sequence.deep_sets.out_size,
            hidden_size=2 * embedding_dim,
            key=gru_key,
        )
        actor_key, critic_key = jax.random.split(key)
        joint_dim = self.env_net.output_size + 2 * config.sequence.embedding_dim
        params = (
            {"num_actions": act_space.n}
            if isinstance(act_space, spaces.Discrete)
            else {"action_dim": act_space.shape[0]}
        )
        self.actor = hydra.utils.instantiate(
            config.actor, in_size=joint_dim, **params, key=actor_key
        )
        self.critic = hydra.utils.instantiate(
            config.critic,
            in_size=joint_dim,
            out_size=1,
            final_layer_activation=False,
            key=critic_key,
        )

    @override
    def _get_action(
        self, features: jax.Array, epsilon_mask: jax.Array
    ) -> distrax.Distribution:
        return self.actor(features, epsilon_mask)

    @override
    def _get_value(self, features: jax.Array) -> jax.Array:
        value = jax.vmap(self.critic)(features)
        return value.squeeze(-1)

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = (
            self.flatten_features(obs.features)
            if self._flatten_features
            else obs.features.features
        )
        x = jax.vmap(self.env_net)(x)
        emb = jax.vmap(self._compute_sequence_embedding)(obs.seq)
        return jnp.concatenate([x, emb], axis=-1)

    def _compute_sequence_embedding(self, seq: JaxReachAvoidSequence) -> jax.Array:
        def embed_assignment_set(indices: jax.Array) -> jax.Array:
            # indices shape: (num_assignments,)
            mask = indices != -1
            embeddings = jax.vmap(self.embedding)(indices * mask) * (mask[:, None])
            # embeddings shape: (num_assignments, embedding_dim)
            return self.deep_sets(embeddings)  # shape: (out_size,)

        reach_emb = jax.vmap(embed_assignment_set)(seq.reach)
        avoid_emb = jax.vmap(embed_assignment_set)(seq.avoid)
        reach_avoid = jnp.concatenate([reach_emb, avoid_emb], axis=-1)
        h0 = jnp.zeros((self.gru.hidden_size,))  # initial hidden state

        def gru_step(
            carry: tuple[jax.Array, int], inputs: jax.Array
        ) -> tuple[tuple[jax.Array, int], None]:
            hidden, step = carry
            hidden = jax.lax.cond(
                step <= seq.depth, lambda: self.gru(inputs, hidden), lambda: hidden
            )
            return (hidden, step - 1), None

        max_seq_length = reach_avoid.shape[0]
        (final_hidden, _), _ = jax.lax.scan(
            gru_step, (h0, max_seq_length), reach_avoid, reverse=True, unroll=8
        )
        return final_hidden

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
