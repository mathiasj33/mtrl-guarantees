import itertools

import hydra
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from rlg import RUN_DIR
from rlg.experiments.simple_grid import WorldParams

sns.set_theme(style="darkgrid")


@hydra.main(version_base="1.1", config_path="../conf", config_name="simple_grid")
def main(cfg: DictConfig):
    params: WorldParams = hydra.utils.instantiate(cfg.world_params)
    dist = params.slip_dist
    xs = np.linspace(0, 0.3, 1000)
    pdf = dist.pdf(xs)

    path = RUN_DIR / "simple_grid" / "eval" / "main"
    guarantees = pd.read_csv(path / "guarantees.csv")
    actual_guarantees = pd.read_csv(path / "actual_guarantees.csv")
    num_tasks = [200]
    num_episodes = [500, 1000]

    colors = sns.color_palette()
    for t, e in itertools.product(num_tasks, num_episodes):
        subset = guarantees[
            (guarantees["num_tasks"] == t) & (guarantees["num_episodes"] == e)
        ]
        print(subset)
        plt.plot(
            subset["guarantee"],
            subset["prob"],
            label=f"{t}/{e} tasks/episodes",
            color=colors.pop(0),
            drawstyle="steps-pre",
        )
    plt.plot(
        actual_guarantees["guarantee"],
        actual_guarantees["actual_safety"],
        label="Actual Safety",
        linestyle="--",
        color=colors[0],
    )
    plt.legend(loc="lower right")
    plt.xlabel("Bound")
    plt.ylabel("Safety")
    plt.show()


if __name__ == "__main__":
    main()
