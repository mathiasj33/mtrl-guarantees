from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from rlg.bounds.expected_performance import compute_guarantees
from rlg.experiments import simple_grid
from rlg.experiments.simple_grid import WorldParams
from rlg.stats.confidence import clopper_pearson


# --- Warm "Anthropic-ish" palette (reds / oranges) ---
ANTHROPIC = {
    "red":    "#C2410C",  # warm red-orange
    "orange": "#EA580C",  # orange
    "ink":    "#111827",  # near-black
    "grid":   "#d1d5db",  # light gray
}


def setup_latex_fonts() -> None:
    """
    Tries to use LaTeX for text rendering. If LaTeX isn't installed, it falls
    back to Matplotlib's serif fonts (still LaTeX-ish).
    """
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                # LaTeX defaults to Computer Modern; keep it explicit:
                "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
                "axes.unicode_minus": False,
            }
        )
    except Exception:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "axes.unicode_minus": False,
            }
        )


def apply_clean_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    # light dashed major grid only
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=1.0,
        color=ANTHROPIC["grid"],
        alpha=0.9,
    )
    ax.grid(False, which="minor")

    # show all four spines + make them thicker
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(ANTHROPIC["ink"])
        ax.spines[side].set_linewidth(1.8)

    # thicker ticks
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        width=1.6,
        length=6,
        colors=ANTHROPIC["ink"],
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.2,
        length=3,
        colors=ANTHROPIC["ink"],
    )


def compute_actual_guarantees(
    params: WorldParams, start: float
) -> tuple[np.ndarray, np.ndarray]:
    guarantees = np.linspace(start, 1.0, 100)
    associated_params = 1 - guarantees ** (1 / (params.width - 1))
    probs: np.ndarray = params.slip_dist.cdf(associated_params)  # type: ignore
    return guarantees, probs


def ensure_guarantee_data(
    cfg: DictConfig, params: WorldParams, output_dir: Path
) -> tuple[Path, Path]:
    guarantees_path = output_dir / "guarantees.csv"
    actual_path = output_dir / "actual_guarantees.csv"
    if guarantees_path.exists() and actual_path.exists():
        return guarantees_path, actual_path

    df = simple_grid.run(
        params=params, num_tasks=int(cfg.num_tasks), num_episodes=int(cfg.num_episodes)
    )
    df.to_csv(output_dir / "results.csv", index=False)

    lower_bounds = clopper_pearson(
        df["num_successes"], df["num_episodes"], cfg.bounds.gamma
    )
    guarantees, probs = compute_guarantees(
        lower_bounds=lower_bounds.tolist(),
        gamma=cfg.bounds.gamma,
        eta=cfg.bounds.eta,
        step_size=cfg.bounds.step_size,
        n_jobs=cfg.bounds.n_jobs,
    )

    pd.DataFrame({"guarantees": guarantees, "probs": probs}).to_csv(
        guarantees_path, index=False
    )

    lowest_guarantee = guarantees[0]
    actual_guarantees, actual_probs = compute_actual_guarantees(
        params, start=lowest_guarantee
    )
    pd.DataFrame({"guarantees": actual_guarantees, "probs": actual_probs}).to_csv(
        actual_path, index=False
    )

    return guarantees_path, actual_path


@hydra.main(version_base="1.1", config_path="../../conf", config_name="simple_grid")
def main(cfg: DictConfig):
    setup_latex_fonts()

    params: WorldParams = hydra.utils.instantiate(cfg.world_params)

    # --- PDF ---
    dist = params.slip_dist
    xs = np.linspace(0, 0.3, 1000)
    pdf = dist.pdf(xs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # Plot 1: slip pdf
    ax0 = axes[0]
    apply_clean_style(ax0)
    ax0.plot(xs, pdf, color=ANTHROPIC["red"], linewidth=3.0)
    ax0.set_title(r"\textbf{Slip Probability Density Function}", fontsize=14, color=ANTHROPIC["ink"])
    ax0.set_xlabel(r"\textbf{Slip Probability}", fontsize=12, color=ANTHROPIC["ink"])
    ax0.set_ylabel(r"\textbf{Density}", fontsize=12, color=ANTHROPIC["ink"])

    # --- Guarantees data ---
    guarantees_path, actual_path = ensure_guarantee_data(cfg, params, Path.cwd())
    guarantees = pd.read_csv(guarantees_path)
    actual_guarantees = pd.read_csv(actual_path)

    # Plot 2: performance guarantees (SWAP AXES)
    ax1 = axes[1]
    apply_clean_style(ax1)

    # swapped: x = guarantees (t), y = probs (1-δ)
    ax1.plot(
        guarantees["guarantees"],
        guarantees["probs"],
        color=ANTHROPIC["red"],
        linewidth=3.0,
        label=r"\textbf{Computed Guarantees}",
    )
    ax1.plot(
        actual_guarantees["guarantees"],
        actual_guarantees["probs"],
        color=ANTHROPIC["orange"],
        linewidth=3.0,
        linestyle="--",
        label=r"\textbf{Actual Guarantees}",
    )

    ax1.set_title(r"\textbf{Performance Guarantees}", fontsize=14, color=ANTHROPIC["ink"])
    ax1.set_xlabel(r"\textbf{Performance Guarantee }$t$", fontsize=12, color=ANTHROPIC["ink"])
    ax1.set_ylabel(r"\textbf{Satisfaction Probability }$(1-\delta)$", fontsize=12, color=ANTHROPIC["ink"])
    ax1.legend(frameon=False, loc="best", fontsize=11, handlelength=3.2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()