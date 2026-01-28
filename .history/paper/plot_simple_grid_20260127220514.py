from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from rlg import RUN_DIR
from rlg.experiments.simple_grid import WorldParams
from rlg.plotting.utils import ANTHROPIC, apply_anthropic_style, setup_latex_fonts


@hydra.main(version_base="1.1", config_path="../conf", config_name="simple_grid")
def main(cfg: DictConfig):
    setup_latex_fonts()

    params: WorldParams = hydra.utils.instantiate(cfg.world_params)

    # --- PDF ---
    dist = params.slip_dist
    xs = np.linspace(0, 0.2, 1000)
    pdf = dist.pdf(xs)

    # ===== Plot 1: Slip Probability (Quadratic) =====
    fig1, ax0 = plt.subplots(figsize=(6, 6), dpi=300)
    fig1.patch.set_facecolor("white")
    apply_anthropic_style(ax0)
    ax0.plot(
        xs,
        pdf,
        color=ANTHROPIC["red_dark"],
        linewidth=2.8,
        solid_capstyle="round",
    )
    ax0.set_xlabel(
        r"\textbf{Slip Probability $p$}",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )
    ax0.set_ylabel(
        r"\textbf{Density}",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )

    fig1.tight_layout(pad=0.8)
    output_path1 = Path.cwd() / "plot_slip_probability.pdf"
    plt.savefig(output_path1, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 1 saved to: {output_path1}")
    plt.close(fig1)

    # ===== Plot 2: Performance Guarantees (Quadratic) =====
    path = RUN_DIR / "simple_grid" / "eval" / "main"
    guarantees = pd.read_csv(path / "guarantees.csv")
    actual_guarantees = pd.read_csv(path / "actual_guarantees.csv")
    num_tasks = 50
    num_episodes = 100
    guarantees = guarantees.query(
        "num_tasks == @num_tasks and num_episodes == @num_episodes"
    )

    fig2, ax1 = plt.subplots(figsize=(6, 6), dpi=300)
    fig2.patch.set_facecolor("white")
    apply_anthropic_style(ax1)

    # swapped: x = guarantees (t), y = probs (1-δ)
    ax1.plot(
        guarantees["guarantee"],
        guarantees["prob"],
        color=ANTHROPIC["red_dark"],
        linewidth=2.8,
        solid_capstyle="round",
        drawstyle="steps-pre",
        label=r"\textbf{Certified Bound $1-\varepsilon$}",
    )
    ax1.plot(
        actual_guarantees["guarantee"],
        actual_guarantees["actual_safety"],
        color=ANTHROPIC["orange_light"],
        linewidth=2.8,
        linestyle="--",
        dash_capstyle="round",
        label=r"\textbf{Actual Safety $S_{\mathcal{D}}^\pi(B)$}",
    )

    ax1.set_xlabel(
        r"\textbf{Performance Threshold }$B$",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )
    ax1.set_ylabel(
        r"\textbf{Safety}",
        fontsize=28,
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

    ax1.set_xlim(left=0.5)
    fig2.tight_layout(pad=0.8)

    # Save as high-quality PDF
    output_path2 = Path.cwd() / "plot_guarantees.pdf"
    plt.savefig(output_path2, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 2 saved to: {output_path2}")

    # plt.show()


if __name__ == "__main__":
    main()
