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

# --- Warm "Anthropic-ish" palette (high contrast pair) ---
ANTHROPIC = {
    "red_dark": "#9A3412",      # deep warm red-brown
    "orange_light": "#FDBA74",  # light peach/orange
    "ink": "#0f172a",           # slate/near-black
    "grid": "#e5e7eb",          # very light gray
}


def setup_latex_fonts() -> None:
    """
    Use LaTeX fonts if available; otherwise fall back to a CM-like look.
    This avoids runtime failures when a full LaTeX install isn't present.
    """
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
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


def apply_anthropic_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    # subtle dashed major grid only
    ax.grid(
        True,
        which="major",
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        color=ANTHROPIC["grid"],
        alpha=0.9,
    )
    ax.grid(False, which="minor")

    # show all spines, thicker
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(ANTHROPIC["ink"])
        ax.spines[side].set_linewidth(1.8)

    # ticks
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
        width=1.6,
        length=6,
        colors=ANTHROPIC["ink"],
        pad=3,
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

    # ===== Plot 1: Slip Probability (Quadratic) =====
    fig1, ax0 = plt.subplots(figsize=(6, 6), dpi=300)
    fig1.patch.set_facecolor("white")
    apply_anthropic_style(ax0)
    ax0.plot(
        xs,
        pdf,
        color=ANTHROPIC["red_dark"],
        linewidth=2.4,
        solid_capstyle="round",
    )
    ax0.set_xlabel(
        r"\textbf{Slip Probability}",
        fontsize=21,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )
    ax0.set_ylabel(
        r"\textbf{Density}",
        fontsize=21,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )

    fig1.tight_layout(pad=0.8)
    output_path1 = Path.cwd() / "plot_slip_probability.pdf"
    plt.savefig(output_path1, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 1 saved to: {output_path1}")
    plt.close(fig1)

    # ===== Plot 2: Performance Guarantees (Quadratic) =====
    guarantees_path, actual_path = ensure_guarantee_data(cfg, params, Path.cwd())
    guarantees = pd.read_csv(guarantees_path)
    actual_guarantees = pd.read_csv(actual_path)

    fig2, ax1 = plt.subplots(figsize=(6, 6), dpi=300)
    fig2.patch.set_facecolor("white")
    apply_anthropic_style(ax1)

    # swapped: x = guarantees (t), y = probs (1-δ)
    ax1.plot(
        guarantees["guarantees"],
        guarantees["probs"],
        color=ANTHROPIC["red_dark"],
        linewidth=2.4,
        solid_capstyle="round",
        label=r"\textbf{Certified Bound $1-\varepsilon$}",
    )
    ax1.plot(
        actual_guarantees["guarantees"],
        actual_guarantees["probs"],
        color=ANTHROPIC["orange_light"],
        linewidth=2.4,
        linestyle="--",
        dash_capstyle="round",
        label=r"\textbf{Actual Safety $S_{\mathcal{D}}^\pi(B)$}",
    )

    ax1.set_xlabel(
        r"\textbf{Performance Threshold }$B$",
        fontsize=21,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )
    ax1.set_ylabel(
        r"\textbf{Safety}",
        fontsize=21,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )

    ax1.legend(
        frameon=False,
        loc="lower left",
        fontsize=16,
        handlelength=3.0,
        labelspacing=0.8,
        borderpad=0.2,
    )

    fig2.tight_layout(pad=0.8)
    
    # Save as high-quality PDF
    output_path2 = Path.cwd() / "plot_guarantees.pdf"
    plt.savefig(output_path2, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 2 saved to: {output_path2}")
    
    #plt.show()


if __name__ == "__main__":
    main()