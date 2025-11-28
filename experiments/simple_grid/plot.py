import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from rlg.experiments.simple_grid import WorldParams

sns.set_theme(style="darkgrid")


@hydra.main(version_base="1.1", config_path="../../conf", config_name="simple_grid")
def main(cfg: DictConfig):
    params: WorldParams = hydra.utils.instantiate(cfg.world_params)
    dist = params.slip_dist
    xs = np.linspace(0, 0.3, 1000)
    pdf = dist.pdf(xs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(xs, pdf)
    axes[0].set_title("Slip Probability Density Function")
    axes[0].set_xlabel("Slip Probability")
    axes[0].set_ylabel("Density")

    guarantees = pd.read_csv("guarantees.csv")
    actual_guarantees = pd.read_csv("actual_guarantees.csv")
    sns.lineplot(
        guarantees, x="probs", y="guarantees", ax=axes[1], label="Computed Guarantees"
    )
    sns.lineplot(
        actual_guarantees,
        x="probs",
        y="guarantees",
        ax=axes[1],
        label="Actual Guarantees",
    )
    axes[1].set_title("Performance Guarantees")
    axes[1].set_xlabel("Satisfaction Probability (1-δ)")
    axes[1].set_ylabel("Performance Guarantee (t)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
