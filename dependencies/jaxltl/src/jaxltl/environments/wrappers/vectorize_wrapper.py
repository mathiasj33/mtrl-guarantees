from functools import partial
from typing import Any, NamedTuple

import jax

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class VectorizeWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]):
    """Vectorize the environment using vmap."""

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, TResetOptions]
        ),
    ):
        super().__init__(env)

    # We currently don't vmap over options at this level (options here are global for
    # all parallel envs)
    # Instead, each env's options get overridden by the env sampler at a lower wrapper
    @partial(jax.vmap, in_axes=(None, 0, 0, None, None))
    def reset(
        self,
        key: jax.Array,
        state: WrapperState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        return super().reset(key, state, params, options)

    @partial(jax.vmap, in_axes=(None, 0, 0, None, None))
    def cheap_reset(
        self,
        key: jax.Array,
        state: WrapperState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        return super().cheap_reset(key, state, params, options)

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, None))
    def step(
        self,
        key: jax.Array,
        state: WrapperState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[WrapperState, TObsFeatures]:
        return super().step(key, state, action, params)
