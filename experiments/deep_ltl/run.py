"""Script to reproduce the DeepLTL experiments in the paper."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from rlg.bounds.expected_performance import compute_guarantees
from rlg.experiments.deep_ltl import deep_ltl
from rlg.stats.confidence import clopper_pearson

logger = logging.getLogger(__name__)

# TODO: add resets to git
# TODO: describe rabinizer dependencies jaxltl


@hydra.main(version_base="1.1", config_path="../../conf", config_name="deep_ltl")
def main(cfg: DictConfig):
    logger.info(
        f"Running DeepLTL experiment with {cfg.num_tasks} tasks and {cfg.num_episodes} episodes per task"
    )
    df = deep_ltl.run(cfg)
    df["tasks_episodes"] = f"{cfg.num_tasks}_{cfg.num_episodes}"
    df.to_csv("results.csv", index=False, mode="a")
    logger.info("Saved results to results.csv")
    logger.info(
        f"Computing guarantees with gamma={cfg.bounds.gamma}, eta={cfg.bounds.eta}"
    )
    lower_bounds = clopper_pearson(
        df["num_successes"], df["num_episodes"], cfg.bounds.gamma
    )
    guarantees, probs = compute_guarantees(
        lower_bounds=lower_bounds.tolist(),
        delta=cfg.bounds.gamma,
        eps=cfg.bounds.eta,
        step_size=cfg.bounds.step_size,
        n_jobs=cfg.bounds.n_jobs,
    )
    df = pd.DataFrame({"guarantees": guarantees, "probs": probs})
    df["tasks_episodes"] = f"{cfg.num_tasks}_{cfg.num_episodes}"
    guarantees_file = Path("guarantees.csv")
    df.to_csv(
        guarantees_file,
        index=False,
        mode="a",
        header=not guarantees_file.exists(),
    )
    logger.info("Saved computed bounds to guarantees.csv")


if __name__ == "__main__":
    main()
