from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl import eqx_utils
from jaxltl.environments.environment import Environment, EnvObservation
from jaxltl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class PrecomputedResetWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]):
    """Environment wrapper that resets to precomputed states loaded from disk. For environments
    with an expensive reset function, this can significantly speed up training. See scripts/precompute_resets.py
    for a script to precompute and save reset states."""

    reset_states: WrapperState  # batched reset states
    num_reset_states: int

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, TResetOptions]
        ),
        params: TEnvParams,
        path: str | Path,
    ):
        super().__init__(env)
        self.num_reset_states = eqx_utils.load_metadata(path)["batch_dim"]
        state_template, _ = env.reset(jax.random.key(0), None, params=params)
        state_template = jax.tree.map(  # Add batch dimension
            lambda x: jnp.zeros((self.num_reset_states,) + x.shape, dtype=x.dtype),
            state_template,
        )
        self.reset_states = eqx_utils.load(path, state_template)

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: WrapperState | None,
        params: TEnvParams,
        options: (
            TResetOptions | None
        ) = None,  # note: currently ignored for precomputed resets
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        state = self._sample_random_reset_state(key)
        obs = self._env.compute_obs(state, params)
        return state, obs

    def _sample_random_reset_state(self, key: jax.Array) -> WrapperState:
        index = jax.random.randint(
            key, shape=(), minval=0, maxval=self.num_reset_states
        )
        return jax.tree.map(lambda x: x[index], self.reset_states)

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: WrapperState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        return self.reset(key, state, params, options)
