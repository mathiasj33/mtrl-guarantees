"""Script to reproduce the simple gridworld motivating experiment in the paper."""

import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from rlg import RUN_DIR
from rlg.experiments import simple_grid
from rlg.experiments.simple_grid import WorldParams

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
    path = RUN_DIR / "simple_grid" / "eval" / "main" / "episode_returns.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info(f"Saved {len(df)} episode results to {path}")

    actual_guarantees, actual_probs = compute_actual_guarantees(params, start=0.0)
    pd.DataFrame(
        {"guarantee": actual_guarantees, "actual_safety": actual_probs}
    ).to_csv(path.parent / "actual_guarantees.csv", index=False)


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
    guarantees = np.linspace(start, 1.0, 1000)
    associated_params = 1 - guarantees ** (1 / (params.width - 1))
    probs: np.ndarray = params.slip_dist.cdf(associated_params)  # type: ignore
    return guarantees, probs


if __name__ == "__main__":
    main()
