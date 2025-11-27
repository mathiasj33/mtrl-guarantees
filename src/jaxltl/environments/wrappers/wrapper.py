"""Environment wrappers.

Adapted from gymnax
(https://github.com/RobertTLange/gymnax/blob/main/gymnax/wrappers/purerl.py).
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition


class WrapperState(eqx.Module):
    """Base class for wrapper states. Wrappers can add fields to this class to maintain
    their own state."""

    state: eqx.Module  # the state of the wrapped environment / previous wrapper

    def unwrapped(self) -> eqx.Module:
        """Recursively unwraps the environment to get the base environment."""
        if isinstance(self.state, WrapperState):
            return self.state.unwrapped()
        return self.state

    def __getattr__(self, name):
        return getattr(self.state, name)


class EnvWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](eqx.Module):
    """Base class for environment wrappers."""

    _env: "EnvWrapper[TEnvParams, TObsFeatures, TResetOptions] | Environment[Any, TEnvParams, TObsFeatures, TResetOptions]"

    def __init__(
        self,
        env: "EnvWrapper[TEnvParams, TObsFeatures, TResetOptions] | Environment[Any, TEnvParams, TObsFeatures, TResetOptions]",
    ):
        self._env = env

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: WrapperState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        return self._env.reset(key, state, params, options)

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: WrapperState,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        return self._env.cheap_reset(key, state, params, options)

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: WrapperState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[WrapperState, TObsFeatures]:
        return self._env.step(key, state, action, params)

    # provide proxy access to regular attributes of wrapped environment
    def __getattr__(self, name):
        return getattr(self._env, name)
