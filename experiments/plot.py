from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

sns.set_theme(style="darkgrid")


@hydra.main(version_base="1.1", config_path="../conf", config_name="plot")
def main(cfg: DictConfig):
    results_dir = Path(cfg.results.dir).resolve()
    guarantees = pd.read_csv(results_dir / "guarantees.csv")
    empirical_safety = pd.read_csv(results_dir / "empirical_safety.csv")
    guarantees["tasks/episodes"] = guarantees.apply(
        lambda row: f"{row['num_tasks']}/{row['num_episodes']}", axis=1
    )
    guarantees = guarantees.query(
        "method in @cfg.plot.bounds and num_tasks in @cfg.plot.num_tasks and num_episodes in @cfg.plot.num_episodes"
    )
    ax = sns.lineplot(
        guarantees,
        x="guarantee",
        y="prob",
        hue="method",
        style="tasks/episodes",
        errorbar=None,
        estimator=None,
        drawstyle="steps-pre",
    )
    sns.lineplot(
        empirical_safety,
        x="guarantee",
        y="empirical_safety",
        errorbar=None,
        estimator=None,
        drawstyle="steps-pre",
        ax=ax,
        linestyle="--",
        label="Empirical Safety",
    )
    ax.set_title("Performance Guarantees")
    ax.set_xlabel("Performance Guarantee (B)")
    ax.set_ylabel("Safety (1-ε)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
