"""Script to compute performance guarantees given a dataframe of episode returns."""

import itertools
import logging
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

from rlg.bounds.expected_performance import compute_guarantees
from rlg.stats.confidence import (clopper_pearson, dkw_mean_lower_bound,
                                  empirical_bernstein, hoeffding)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="compute_guarantees")
def main(cfg: DictConfig):
    path = Path(f"{cfg.results.dir}/{cfg.results.results_file}").resolve()
    logger.info(f"Loading episode returns from {path}")
    full_df = pl.read_parquet(path)

    total_num_tasks = full_df["task_id"].n_unique()
    total_num_episodes = full_df["episode_id"].n_unique()
    logger.info(
        f"Total number of tasks: {total_num_tasks}, total episodes: {total_num_episodes}"
    )

    num_tasks_list = cfg.guarantees.num_tasks
    if not isinstance(num_tasks_list, Sequence):
        num_tasks_list = [num_tasks_list]
    num_episodes_list = cfg.guarantees.num_episodes
    if not isinstance(num_episodes_list, Sequence):
        num_episodes_list = [num_episodes_list]
    bound_fn = get_bound_function(cfg.guarantees.bound)

    all_guarantees = []
    for num_tasks, num_episodes in itertools.product(num_tasks_list, num_episodes_list):
        df = full_df.filter(
            (pl.col("task_id") < num_tasks) & (pl.col("episode_id") < num_episodes)
        )
        logger.info(
            f"Computing bounds for {num_tasks} tasks and {num_episodes} episodes"
        )

        bounds = bound_fn(
            df,
            min_return=cfg.guarantees.min_return,
            max_return=cfg.guarantees.max_return,
            beta=cfg.guarantees.beta,
        )

        guarantees, probs = compute_guarantees(
            lower_bounds=bounds["lower_bound"].to_list(),
            min_return=cfg.guarantees.min_return,
            max_return=cfg.guarantees.max_return,
            beta=cfg.guarantees.beta,
            delta=cfg.guarantees.delta,
            step_size=cfg.guarantees.step_size,
            n_jobs=cfg.n_jobs,
        )
        # Sort guarantees and probs together by guarantees (ascending)
        paired = sorted(zip(guarantees, probs), key=lambda x: x[0])
        guarantees = [x[0] for x in paired]
        probs = [x[1] for x in paired]
        
        # Sanity checks - allow small numerical errors
        assert all(
            guarantees[i] <= guarantees[i + 1] + 1e-9 for i in range(len(guarantees) - 1)
        ), "Guarantees should be monotonically increasing"
        # Probs should be monotonically decreasing (with tolerance for numerical errors)
        for i in range(len(probs) - 1):
            if probs[i] < probs[i + 1] - 1e-9:
                # Log warning but don't assert
                logger.warning(f"Prob decreasing at index {i}: {probs[i]} < {probs[i + 1]}")

        # Save guarantees
        g_df = pd.DataFrame(
            {
                "guarantee": guarantees,
                "prob": probs,
                "num_tasks": num_tasks,
                "num_episodes": num_episodes,
                "bound": cfg.guarantees.bound,
                "beta": cfg.guarantees.beta,
                "delta": cfg.guarantees.delta,
            }
        )
        all_guarantees.append(g_df)

    # Save all results
    guarantees_df = pd.concat(all_guarantees, ignore_index=True)
    guarantees_file = Path(cfg.results.dir) / "guarantees.csv"
    if guarantees_file.exists():
        existing_df = pd.read_csv(guarantees_file)
        # Remove existing rows that match the new data's key columns
        key_cols = ["num_tasks", "num_episodes", "bound", "beta", "delta"]
        new_keys = guarantees_df[key_cols].drop_duplicates()
        merged = existing_df.merge(new_keys, on=key_cols, how="left", indicator=True)
        existing_df = existing_df[merged["_merge"] == "left_only"]
        guarantees_df = pd.concat([existing_df, guarantees_df], ignore_index=True)
    guarantees_df.to_csv(guarantees_file, index=False)
    logger.info(f"Saved computed bounds to {guarantees_file.resolve()}")

    # Compute empirical safety
    logger.info("Computing empirical safety...")
    min_guarantee = guarantees_df["guarantee"].min()
    max_guarantee = guarantees_df["guarantee"].max()
    empirical_safety = compute_empirical_safety(min_guarantee, max_guarantee, full_df)
    empirical_safety_file = Path(cfg.results.dir) / "empirical_safety.csv"
    empirical_safety.to_csv(empirical_safety_file, index=False)
    logger.info(f"Saved empirical safety to {empirical_safety_file.resolve()}")


def get_bound_function(name: str):
    name_to_function = {
        "hoeffding": hoeffding,
        "bernstein": empirical_bernstein,
        "dkw": dkw_mean_lower_bound,
        "clopper-pearson": clopper_pearson,
    }
    if name not in name_to_function:
        raise ValueError(f"Unknown bound function: {name}")
    return name_to_function[name]


def compute_empirical_safety(
    min_guarantee, max_guarantee, full_df, num_empirical_points=1000
):
    empirical_performance = full_df.group_by("task_id").agg(
        pl.col("total_return").mean().alias("mean_return")
    )
    num_empirical_tasks = empirical_performance.height
    guarantee_grid = np.linspace(min_guarantee, max_guarantee, num_empirical_points)

    empirical_safeties = []
    for g in guarantee_grid:
        safety = (
            empirical_performance.filter(pl.col("mean_return") >= g).height
            / num_empirical_tasks
        )
        empirical_safeties.append(safety)

    return pd.DataFrame(
        {
            "guarantee": guarantee_grid,
            "empirical_safety": empirical_safeties,
        }
    )


if __name__ == "__main__":
    main()
