import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

sns.set_theme(style="darkgrid")


@hydra.main(version_base="1.1", config_path="../../conf", config_name="deep_ltl")
def main(cfg: DictConfig):
    guarantees = pd.read_csv("guarantees.csv")
    ax = sns.lineplot(
        guarantees,
        x="probs",
        y="guarantees",
        hue="tasks_episodes",
        errorbar=None,
        estimator=None,
    )
    # ax.set_xlim(0.8, 1.0)
    ax.set_title("Performance Guarantees")
    ax.set_xlabel("Satisfaction Probability (1-δ)")
    ax.set_ylabel("Performance Guarantee (t)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
