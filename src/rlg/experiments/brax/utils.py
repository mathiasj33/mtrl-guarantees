"""Utility functions for Brax experiments."""

from pathlib import Path
from typing import NamedTuple

import jax

from rlg.experiments.brax.brax_multi_task_wrapper import TaskParams
from rlg.experiments.brax.cheetah_robust import CheetahRobust
from rlg.experiments.brax.walker_robust import WalkerRobust

_env_registry = {"walker": WalkerRobust, "cheetah": CheetahRobust}


def find_latest_checkpoint(base_path: Path) -> Path:
    """Find the latest checkpoint in the given directory."""
    checkpoints = [f for f in base_path.glob("*") if f.is_dir()]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {base_path}")
    return max(checkpoints, key=lambda f: int(f.name))


def load_env(name: str) -> WalkerRobust | CheetahRobust:
    """Loads the specified Brax environment by name."""
    if name not in _env_registry:
        raise ValueError(f"Environment '{name}' not found in registry.")
    return _env_registry[name]()


def sample_task(rng: jax.Array, log_tau_min: float, log_tau_max: float) -> NamedTuple:
    """Samples task parameters from the RoML distribution."""
    log_taus = jax.random.uniform(
        rng, shape=(2,), minval=log_tau_min, maxval=log_tau_max
    )
    taus = 2.0**log_taus
    return TaskParams(mass_scale=taus[0], length_scale=taus[1])
