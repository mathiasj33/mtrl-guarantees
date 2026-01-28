"""Script to reproduce the simple gridworld motivating experiment in the paper."""

import logging

import hydra
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

from rlg.bounds.expected_performance import compute_guarantees
from rlg.experiments import simple_grid
from rlg.experiments.simple_grid import WorldParams
from rlg.stats.confidence import clopper_pearson

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="simple_grid")
def main(cfg: DictConfig):
    params: WorldParams = hydra.utils.instantiate(cfg.world_params)
    logger.info(
        f"Running simple gridworld experiment with {cfg.num_tasks} tasks and {cfg.num_episodes} episodes per task"
    )
    df = simple_grid.run(
        params=params, num_tasks=int(cfg.num_tasks), num_episodes=int(cfg.num_episodes)
    )
    df.to_csv("results.csv", index=False)
    logger.info("Saved results to results.csv")
    logger.info(
        f"Computing guarantees with gamma={cfg.bounds.gamma}, eta={cfg.bounds.eta}"
    )
    lower_bounds = clopper_pearson(
        df, min_return=0.0, max_return=1.0, beta=cfg.bounds.gamma
    )
    guarantees, probs = compute_guarantees(
        lower_bounds=lower_bounds.tolist(),
        gamma=cfg.bounds.gamma,
        eta=cfg.bounds.eta,
        step_size=cfg.bounds.step_size,
        n_jobs=cfg.bounds.n_jobs,
    )
    pd.DataFrame({"guarantees": guarantees, "probs": probs}).to_csv(
        "guarantees.csv", index=False
    )
    logger.info("Saved computed bounds to guarantees.csv")
    lowest_guarantee = guarantees[0]
    actual_guarantees, actual_probs = compute_actual_guarantees(
        params, start=lowest_guarantee
    )
    pd.DataFrame({"guarantees": actual_guarantees, "probs": actual_probs}).to_csv(
        "actual_guarantees.csv", index=False
    )


def compute_actual_guarantees(
    params: WorldParams, start: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute actual guarantees for the optimal policy in the simple gridworld.

    Args:
        params: WorldParams object defining the gridworld.
        start: The starting performance guarantee to compute from.

    Returns:
        A tuple (guarantees, probs) of performance guarantees and the probability with
        which they are satisfied.
    """
    guarantees = np.linspace(start, 1.0, 100)
    associated_params = 1 - guarantees ** (1 / (params.width - 1))
    probs: np.ndarray = params.slip_dist.cdf(associated_params)  # type: ignore
    return guarantees, probs


if __name__ == "__main__":
    main()
