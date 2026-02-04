"""Compute performance guarantees across multiple task batches."""

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

from rlg import DATA_DIR
from rlg.bounds.expected_performance import compute_guarantees
from rlg.plotting.utils import summarize_step_curves
from rlg.stats.confidence import (
    clopper_pearson,
    dkw_mean_lower_bound,
    empirical_bernstein,
    hoeffding,
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1", config_path="../conf", config_name="compute_guarantees"
)
def main(cfg: DictConfig):
    returns_path = DATA_DIR / cfg.env.name / "episode_returns.parquet"
    logger.info(f"Loading episode returns from {returns_path}")
    full_df = pl.read_parquet(returns_path)

    bound_name = cfg.guarantees.bound
    min_return = 0.0
    max_return = 1000.0 if cfg.env.name in ["cheetah", "walker"] else 0.0
    beta = cfg.guarantees.beta
    delta = cfg.guarantees.delta

    if cfg.guarantees.bound == "clopper-pearson" and (
        min_return != 0.0 or max_return != 1.0
    ):
        raise ValueError("Clopper-Pearson bound requires returns in [0, 1] range.")

    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_num_tasks = full_df["task_id"].n_unique()
    total_num_episodes = full_df["episode_id"].n_unique()
    logger.info(
        f"Total number of tasks: {total_num_tasks}, total episodes: {total_num_episodes}"
    )

    num_episodes_list = cfg.guarantees.num_episodes
    if not isinstance(num_episodes_list, Sequence):
        num_episodes_list = [num_episodes_list]
    bound_fn = get_bound_function(bound_name)

    num_tasks = int(cfg.guarantees.num_tasks)
    max_batches = total_num_tasks // num_tasks
    requested_batches = int(cfg.guarantees.num_batches)
    num_batches = min(requested_batches, max_batches)
    if num_batches < requested_batches:
        logger.warning(
            "Requested %s batches of size %s, but only %s are available.",
            requested_batches,
            num_tasks,
            num_batches,
        )

    all_guarantees = []
    for num_episodes in num_episodes_list:
        logger.info(
            "Computing bounds for %s batches with %s tasks and %s episodes",
            num_batches,
            num_tasks,
            num_episodes,
        )
        episode_guarantees = []
        timings = []
        for batch_id in range(num_batches):
            task_start = batch_id * num_tasks
            task_end = task_start + num_tasks
            df = full_df.filter(
                (pl.col("task_id") >= task_start)
                & (pl.col("task_id") < task_end)
                & (pl.col("episode_id") < num_episodes)
            )
            tasks_in_batch = df["task_id"].n_unique()
            if tasks_in_batch == 0:
                logger.warning(
                    "Skipping batch %s (tasks %s-%s): no data found.",
                    batch_id,
                    task_start,
                    task_end - 1,
                )
                continue
            if tasks_in_batch < num_tasks:
                logger.warning(
                    "Batch %s has only %s tasks (expected %s).",
                    batch_id,
                    tasks_in_batch,
                    num_tasks,
                )

            start_time = time.perf_counter()
            bounds = bound_fn(
                df,
                min_return=min_return,
                max_return=max_return,
                beta=beta,
            )

            guarantees, probs = compute_guarantees(
                lower_bounds=bounds["lower_bound"].to_list(),
                min_return=min_return,
                max_return=max_return,
                beta=beta,
                delta=delta,
                step_size=cfg.guarantees.step_size,
                n_jobs=cfg.n_jobs,
            )
            elapsed = time.perf_counter() - start_time

            paired = sorted(zip(guarantees, probs, strict=False), key=lambda x: x[0])
            guarantees = [x[0] for x in paired]
            probs = [x[1] for x in paired]

            assert all(
                guarantees[i] <= guarantees[i + 1] + 1e-9
                for i in range(len(guarantees) - 1)
            ), "Guarantees should be monotonically increasing"
            for i in range(len(probs) - 1):
                if probs[i] < probs[i + 1] - 1e-9:
                    logger.warning(
                        "Prob decreasing at index %s in batch %s: %s < %s",
                        i,
                        batch_id,
                        probs[i],
                        probs[i + 1],
                    )

            g_df = pd.DataFrame(
                {
                    "guarantee": guarantees,
                    "prob": probs,
                    "num_tasks": num_tasks,
                    "num_episodes": num_episodes,
                    "bound": bound_name,
                    "beta": beta,
                    "delta": delta,
                    "batch_id": batch_id,
                    "task_start": task_start,
                    "task_end": task_end - 1,
                    "tasks_in_batch": tasks_in_batch,
                }
            )
            all_guarantees.append(g_df)
            episode_guarantees.append(g_df)
            timings.append(
                {
                    "batch_id": batch_id,
                    "num_tasks": num_tasks,
                    "num_episodes": num_episodes,
                    "seconds": elapsed,
                    "tasks_in_batch": tasks_in_batch,
                }
            )

        if not episode_guarantees:
            logger.warning("No guarantees computed for %s episodes.", num_episodes)
        else:
            guarantees_df = pd.concat(episode_guarantees, ignore_index=True)
            guarantees_file = (
                Path(cfg.output)
                / f"guarantees_tasks{num_tasks}_episodes{num_episodes}.csv"
            )
            if guarantees_file.exists():
                existing_df = pd.read_csv(guarantees_file)
                key_cols = [
                    "num_tasks",
                    "num_episodes",
                    "bound",
                    "beta",
                    "delta",
                ]
                new_keys = guarantees_df[key_cols].drop_duplicates()
                merged = existing_df.merge(
                    new_keys, on=key_cols, how="left", indicator=True
                )
                existing_df = existing_df[merged["_merge"] == "left_only"]
                guarantees_df = pd.concat(
                    [existing_df, guarantees_df], ignore_index=True
                )
            guarantees_df.to_csv(guarantees_file, index=False)
            logger.info(f"Saved batch bounds to {guarantees_file.resolve()}")

            timing_df = pd.DataFrame(timings)
            timing_file = (
                Path(cfg.output) / f"timing_tasks{num_tasks}_episodes{num_episodes}.csv"
            )
            timing_df.to_csv(timing_file, index=False)
            logger.info(f"Saved timing data to {timing_file.resolve()}")

            num_points = cfg.guarantees.get("summary_points", 600)
            grid, mean, std = summarize_step_curves(
                guarantees_df,
                num_tasks=num_tasks,
                num_episodes=num_episodes,
                num_points=num_points,
            )
            time_mean = float(timing_df["seconds"].mean())
            time_std = float(timing_df["seconds"].std(ddof=0) or 0.0)
            summary_df = pd.DataFrame(
                {
                    "guarantee": grid,
                    "prob_mean": mean,
                    "prob_std": std,
                    "num_tasks": num_tasks,
                    "num_episodes": num_episodes,
                    "num_batches": len(timings),
                    "time_mean_s": time_mean,
                    "time_std_s": time_std,
                }
            )
            summary_file = (
                Path(cfg.output)
                / f"guarantees_summary_tasks{num_tasks}_episodes{num_episodes}.csv"
            )
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved summary data to {summary_file.resolve()}")

    if not all_guarantees:
        raise RuntimeError("No guarantees were computed; check batch settings.")
    guarantees_df = pd.concat(all_guarantees, ignore_index=True)

    logger.info("Computing empirical safety...")
    min_guarantee = guarantees_df["guarantee"].min()
    max_guarantee = guarantees_df["guarantee"].max()
    empirical_safety = compute_empirical_safety(min_guarantee, max_guarantee, full_df)
    empirical_safety_file = Path(cfg.output) / "empirical_safety.csv"
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
