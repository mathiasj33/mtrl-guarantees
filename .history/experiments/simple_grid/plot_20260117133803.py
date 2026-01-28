import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from rlg.experiments.simple_grid import WorldParams

# --- Anthropic-ish muted palette (teal + amber) ---
ANTHROPIC = {
    "teal":  "#2A9D8F",
    "amber": "#E9C46A",
    "ink":   "#1a1a1a",
    "grid":  "#c0c0c0",
}

def apply_clean_style(ax):
    ax.set_facecolor("white")
    # light dashed major grid only
    ax.grid(True, which="major", linestyle="--", linewidth=0.8,
            color=ANTHROPIC["grid"], alpha=0.9)
    ax.grid(False, which="minor")

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ANTHROPIC["ink"])
    ax.spines["bottom"].set_color(ANTHROPIC["ink"])
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    ax.tick_params(axis="both", labelsize=12, width=1.1, colors=ANTHROPIC["ink"])


@hydra.main(version_base="1.1", config_path="../../conf", config_name="simple_grid")
def main(cfg: DictConfig):
    params: WorldParams = hydra.utils.instantiate(cfg.world_params)
    dist = params.slip_dist

    xs = np.linspace(0, 0.3, 1000)
    pdf = dist.pdf(xs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # --- Plot 1: slip pdf ---
    ax = axes[0]
    apply_clean_style(ax)
    ax.plot(xs, pdf, color=ANTHROPIC["teal"], linewidth=2.6)
    ax.set_title("Slip Probability Density Function", fontsize=14, fontweight="bold", color=ANTHROPIC["ink"])
    ax.set_xlabel("Slip Probability", fontsize=12, fontweight="bold", color=ANTHROPIC["ink"])
    ax.set_ylabel("Density", fontsize=12, fontweight="bold", color=ANTHROPIC["ink"])

    # --- Plot 2: performance guarantees (SWAP AXES) ---
    guarantees = pd.read_csv("guarantees.csv")
    actual_guarantees = pd.read_csv("actual_guarantees.csv")

    ax = axes[1]
    apply_clean_style(ax)

    # swapped: x = guarantees (t), y = probs (1-δ)
    ax.plot(
        guarantees["guarantees"], guarantees["probs"],
        color=ANTHROPIC["teal"], linewidth=2.6, label="Computed Guarantees"
    )
    ax.plot(
        actual_guarantees["guarantees"], actual_guarantees["probs"],
        color=ANTHROPIC["amber"], linewidth=2.6, linestyle="--", label="Actual Guarantees"
    )

    ax.set_title("Performance Guarantees", fontsize=14, fontweight="bold", color=ANTHROPIC["ink"])
    ax.set_xlabel("Performance Guarantee (t)", fontsize=12, fontweight="bold", color=ANTHROPIC["ink"])
    ax.set_ylabel("Satisfaction Probability (1-δ)", fontsize=12, fontweight="bold", color=ANTHROPIC["ink"])
    ax.legend(frameon=False, loc="best", fontsize=12, handlelength=3.0)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()