"""
Functions for reproducing the simple gridworld motivating experiment in the paper.
"""

from typing import Any, NamedTuple

import numpy as np
import polars as pl
from tqdm import trange


class WorldParams(NamedTuple):
    slip_dist: Any
    width: int


def run(params: WorldParams, num_tasks: int, num_episodes: int) -> pl.DataFrame:
    dfs = []
    for task_id in trange(num_tasks):
        results = collect_episodes(params, num_episodes)
        df = pl.DataFrame(
            {
                "task_id": task_id,
                "episode_id": np.arange(num_episodes),
                "total_return": results,
            }
        )
        dfs.append(df)
    return pl.concat(dfs)


def collect_episodes(params: WorldParams, num_episodes: int) -> list[float]:
    """Run multiple episodes in the simple gridworld with the same slip probability and return success results."""
    slip_prob = params.slip_dist.sample()
    results = []
    for _ in range(num_episodes):
        pos = 0
        while pos < params.width - 1:
            failure = np.random.uniform() < slip_prob
            if failure:
                results.append(0.0)
                break
            pos += 1
        else:
            results.append(1.0)
    return results
