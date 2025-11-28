"""
Functions for reproducing the simple gridworld motivating experiment in the paper.
"""

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from tqdm import trange


class WorldParams(NamedTuple):
    slip_dist: Any
    width: int


def run(params: WorldParams, num_tasks: int, num_episodes: int) -> pd.DataFrame:
    data = []
    for task_id in trange(num_tasks):
        results = collect_episodes(params, num_episodes)
        data.append(
            [task_id, num_episodes, np.sum(results), np.min(results), np.max(results)]
        )

    df = pd.DataFrame(
        data,
        columns=[
            "task_id",
            "num_episodes",
            "num_successes",
            "min",
            "max",
        ],
    )
    return df


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
