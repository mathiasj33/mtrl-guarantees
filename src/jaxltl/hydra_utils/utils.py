from typing import Any

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf


def register_custom_resolvers():
    activation_functions = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
    }
    OmegaConf.register_new_resolver("act", lambda name: activation_functions[name])


def resolve_default_options(env_config: DictConfig) -> Any | None:
    """Resolve the default_options from the environment config.
    Args:
        env_config: DictConfig for the environment.
    Returns:
        NamedTuple of default options, or None if not specified.
    """
    default_options = None
    if "default_options" in env_config:
        # Instantiate the default_options object from config.
        # The fields will be standard python types (e.g., lists).
        default_options_with_lists = hydra.utils.instantiate(env_config.default_options)

        # Convert all leaf elements (the lists) in the pytree to jax arrays.
        default_options = jax.tree.map(
            lambda x: jnp.array(x, dtype=jnp.float32), default_options_with_lists
        )
    return default_options
