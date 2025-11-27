"""Proximal Policy Optimization (PPO) algorithm implementation.

Adapted from PureJaxRL's PPO implementation (https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py).
"""

import math
from collections.abc import Callable
from typing import NamedTuple, override

import equinox as eqx
import jax
import jax.experimental
import jax.numpy as jnp
import optax
from jax.experimental import io_callback
from jaxtyping import PyTree

from jaxltl import eqx_utils
from jaxltl.environments.environment import Environment, EnvParams
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.eqx_utils.training import TrainState
from jaxltl.rl.actor_critic import ActorCritic
from jaxltl.rl.algorithm import RLAlgorithm


class PPOConfig(NamedTuple):
    total_timesteps: int
    num_envs: int
    num_steps: int
    num_minibatches: int
    update_epochs: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    lr: float
    max_grad_norm: float
    anneal_lr: bool
    adam_eps: float


class PPOTransition(NamedTuple):
    """PPO-relevant transition information."""

    terminated: jax.Array
    truncated: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: PyTree
    terminal_obs: PyTree
    info: PyTree


class PPO(RLAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm."""

    config: PPOConfig

    def __init__(self, **kwargs):
        self.config = PPOConfig(**kwargs)
        if (
            self.config.num_envs * self.config.num_steps
        ) % self.config.num_minibatches != 0:
            raise ValueError(
                "num_envs * num_steps (num_transitions) must be divisible by num_minibatches"
            )

    @override
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def train(
        self,
        model: ActorCritic,
        env: Environment | EnvWrapper,
        env_params: EnvParams,
        key: jax.Array,
        callback: Callable | None = None,
        callback_freq: int | None = None,
        seed: jax.Array | None = None,
    ) -> tuple[ActorCritic, dict]:
        """Train the model using PPO."""

        # Initialize optimizer and training state
        optim = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(
                learning_rate=(
                    self.linear_schedule if self.config.anneal_lr else self.config.lr
                ),
                eps=self.config.adam_eps,
            ),
        )
        train_state = TrainState.create(model, optim)

        # Initialize environment
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.config.num_envs)
        env_state, obsv = env.reset(reset_keys, None, env_params, None)

        # Calculate number of updates and callback intervals
        num_steps_per_update = self.config.num_envs * self.config.num_steps
        num_updates = self.config.total_timesteps // num_steps_per_update
        if callback_freq is not None:
            if callback is None:
                raise ValueError("callback_freq is set but callback is None")
            updates_per_callback = math.ceil(callback_freq / num_steps_per_update)
            num_callbacks = math.ceil(num_updates / updates_per_callback)
        else:
            updates_per_callback = num_updates
            num_callbacks = 1

        # Training loop
        def callback_iter(
            carry: tuple[TrainState[ActorCritic], PyTree, PyTree, jax.Array, jax.Array],
            _,
        ):
            def step(
                carry: tuple[
                    TrainState[ActorCritic], PyTree, PyTree, jax.Array, jax.Array
                ],
                _,
            ):
                train_state, obsv, env_state, key, step_count = carry
                key, step_key = jax.random.split(key)
                train_state, obsv, env_state, metric = self._train_step(
                    train_state,
                    optim,
                    obsv,
                    env,
                    env_state,
                    env_params,
                    key=step_key,
                )
                carry = (train_state, obsv, env_state, key, step_count + 1)
                return carry, metric

            carry, metric = eqx_utils.filter_scan(
                step, carry, None, updates_per_callback
            )
            if callback:
                train_state, step_count = carry[0], carry[4]
                total_step = step_count * self.config.num_envs * self.config.num_steps
                params, _ = eqx.partition(train_state.model, eqx.is_array)
                io_callback(callback, None, metric, params, seed, total_step)
            return carry, metric

        key, update_key = jax.random.split(key)
        carry = (train_state, obsv, env_state, update_key, jnp.zeros((), jnp.int32))
        carry, metric = eqx_utils.filter_scan(callback_iter, carry, None, num_callbacks)
        train_state = carry[0]
        return train_state.model, metric

    def linear_schedule(self, count):
        frac = (
            1.0
            - (count // (self.config.num_minibatches * self.config.update_epochs))
            / self.config.total_timesteps
        )
        return self.config.lr * frac

    def _train_step(
        self,
        train_state: TrainState[ActorCritic],
        optim: optax.GradientTransformation,
        obsv: PyTree,
        env: Environment | EnvWrapper,
        env_state: PyTree,
        env_params: PyTree,
        *,
        key: jax.Array,
    ) -> tuple[TrainState[ActorCritic], jax.Array, PyTree, PyTree]:
        """Perform a single PPO train step.

        Returns:
            train_state: The updated training state after the step.
            obsv: The last observation after the rollout.
            env_state: The updated environment state after the rollout.
            metric: Metrics collected during the rollout.
        """

        # Collect trajectories and compute advantages
        key, rollout_key = jax.random.split(key)
        trajs, last_obs, env_state = self._rollout(
            train_state.model,
            obsv,
            env,
            env_state,
            env_params,
            key=rollout_key,
        )

        advantages, targets = self._calculate_gae(trajs, last_obs, train_state.model)

        # Update update_epochs number of times over the collected data
        def update_epoch(carry: tuple[TrainState[ActorCritic], jax.Array], _):
            train_state, key = carry
            key, shuffle_key = jax.random.split(key)
            minibatches = self._get_minibatches(trajs, advantages, targets, shuffle_key)

            def update_minibatch(
                train_state: TrainState[ActorCritic], minibatch: PyTree
            ):
                trajs, advantages, targets = minibatch
                grad_fn = eqx.filter_value_and_grad(self._loss_fn, has_aux=True)
                losses, grads = grad_fn(train_state.model, trajs, advantages, targets)
                train_state = train_state.apply_gradients(optim, grads)
                return train_state, losses

            train_state, losses = eqx_utils.filter_scan(
                update_minibatch, train_state, minibatches
            )
            return (train_state, key), losses

        key, update_key = jax.random.split(key)
        (train_state, key), losses = eqx_utils.filter_scan(
            update_epoch, (train_state, update_key), None, self.config.update_epochs
        )
        metric = trajs.info
        return train_state, last_obs, env_state, metric

    def _rollout(
        self,
        model: ActorCritic,
        obsv: PyTree,
        env: Environment | EnvWrapper,
        env_state: PyTree,
        env_params: PyTree,
        *,
        key: jax.Array,
    ) -> tuple[PPOTransition, PyTree, PyTree]:
        """Collect a batch of trajectories using the current policy.

        Returns:
            trajs: A batch of collected transitions.
            last_obsv: The last observation after the rollout.
            env_state: The updated environment state after the rollout."""

        def env_step(carry: tuple[PyTree, PyTree, jax.Array], _):
            env_state, last_obs, key = carry

            # select action
            key, sample_key = jax.random.split(key)
            pi, value = model(last_obs)
            action, log_prob = pi.sample_and_log_prob(seed=sample_key)

            # step env
            key, step_key = jax.random.split(key)
            step_key = jax.random.split(step_key, self.config.num_envs)
            transition = env.step(step_key, env_state, action, env_params)
            ppo_transition = PPOTransition(
                transition.terminated,
                transition.truncated,
                action,
                value,
                transition.reward,
                log_prob,  # type: ignore
                last_obs,
                transition.terminal_observation,
                transition.info,
            )
            carry = (transition.state, transition.observation, key)
            return carry, ppo_transition

        carry = (env_state, obsv, key)
        carry, trajs = jax.lax.scan(env_step, carry, None, self.config.num_steps)
        return trajs, carry[1], carry[0]

    def _calculate_gae(
        self, trajs: PPOTransition, last_obs: PyTree, model: ActorCritic
    ) -> tuple[jax.Array, jax.Array]:
        """Calculate Generalized Advantage Estimation (GAE) and target values.

        Returns:
            advantages: The calculated advantages.
            targets: The target values for the critic."""

        def get_advantages(
            gae_and_next_value: tuple[jax.Array, jax.Array], transition: PPOTransition
        ):
            gae, next_value = gae_and_next_value
            next_value = jax.lax.select(
                transition.truncated,
                model.get_value(transition.terminal_obs),
                next_value,
            )
            term, value, reward = (
                transition.terminated,
                transition.value,
                transition.reward,
            )
            not_term = 1.0 - term.astype(jnp.float32)
            delta = reward + self.config.gamma * next_value * not_term - value
            gae = delta + self.config.gamma * self.config.gae_lambda * not_term * gae
            return (gae, value), gae

        last_val = jax.lax.select(
            trajs.truncated[-1],
            model.get_value(jax.tree.map(lambda x: x[-1], trajs.terminal_obs)),
            model.get_value(last_obs),
        )
        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajs,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + trajs.value

    def _get_minibatches(
        self,
        trajs: PPOTransition,
        advantages: jax.Array,
        targets: jax.Array,
        key: jax.Array,
    ) -> PyTree:
        """Prepare shuffled minibatches from the collected trajectories.

        Returns:
            minibatches: PyTree with shape (num_minibatches, batch_size_per_minibatch, ...).
        """

        num_transitions = self.config.num_steps * self.config.num_envs
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_transitions)
        data = (trajs, advantages, targets)
        # flatten from (steps, envs, ...) to (steps*envs, ...) = (num_transitions, ...)
        data = jax.tree.map(lambda x: x.reshape((num_transitions,) + x.shape[2:]), data)
        shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), data)
        minibatches = jax.tree.map(
            lambda x: x.reshape((self.config.num_minibatches, -1) + x.shape[1:]),
            shuffled,
        )
        return minibatches

    def _loss_fn(
        self,
        model: ActorCritic,
        trajs: PPOTransition,
        gae: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        """Calculate the PPO loss.

        Returns:
            total_loss: The total loss combining actor and critic losses.
            (value_loss, actor_loss, entropy): A tuple of individual loss components."""
        # re-run model
        pi, value = model(trajs.obs)
        log_prob = pi.log_prob(trajs.action)

        # value loss
        value_pred_clipped = trajs.value + (value - trajs.value).clip(
            -self.config.clip_eps, self.config.clip_eps
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # actor loss
        ratio = jnp.exp(log_prob - trajs.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - self.config.clip_eps,
                1.0 + self.config.clip_eps,
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        # total loss
        total_loss = (
            loss_actor
            + self.config.vf_coef * value_loss
            - self.config.ent_coef * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)
