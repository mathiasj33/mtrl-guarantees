import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rlg import PAPER_DIR, RUN_DIR
from rlg.plotting.utils import (
    ANTHROPIC,
    apply_anthropic_style,
    guarantees_batches_filename,
    plot_suffix_from_combos,
    resolve_plot_combos,
    setup_latex_fonts,
    summarize_step_curves,
)

sns.set_theme(style="darkgrid")


def main():
    setup_latex_fonts()

    combos = resolve_plot_combos(100, 1000)
    results_dir = os.getenv("PLOT_RESULTS_DIR")
    if results_dir:
        base_dir = Path(results_dir)
    else:
        base_dir = RUN_DIR / "ZoneEnv/eval/main"
    empirical = pd.read_csv(base_dir / "empirical_safety.csv")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("white")
    apply_anthropic_style(ax)

    colors = plt.cm.Reds(np.linspace(0.45, 0.85, len(combos)))
    for (num_tasks, num_episodes), color in zip(combos, colors):
        guarantees = pd.read_csv(
            base_dir / guarantees_batches_filename(num_tasks, num_episodes)
        )
        grid, mean, std = summarize_step_curves(
            guarantees, num_tasks=num_tasks, num_episodes=num_episodes, num_points=600
        )
        band_low = np.clip(mean - 1.5 * std, 0.0, 1.0)
        band_high = np.clip(mean + 1.5 * std, 0.0, 1.0)
        ax.fill_between(
            grid,
            band_low,
            band_high,
            color=color,
            alpha=0.22,
            step="pre",
        )
        ax.plot(
            grid,
            mean,
            color=color,
            linewidth=2.8,
            solid_capstyle="round",
            drawstyle="steps-pre",
            label=rf"\textbf{{Certified Bound ({num_tasks}, {num_episodes})}}",
        )
    ax.plot(
        empirical["guarantee"],
        empirical["empirical_safety"],
        color=ANTHROPIC["red_dark"],
        linewidth=3.2,
        linestyle="--",
        dash_capstyle="round",
        label=r"\textbf{Empirical Safety}",
    )

    ax.set_xlabel(
        r"\textbf{Performance Threshold }$B$",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )
    ax.set_ylabel(
        r"\textbf{Safety}",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )

    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=16,
        handlelength=3.0,
        labelspacing=0.8,
        borderpad=0.2,
    )

    ax.set_xlim(left=0.0)
    fig.tight_layout(pad=0.8)
    suffix = plot_suffix_from_combos(combos)
    output_dir = os.getenv("PLOT_OUTPUT_DIR")
    out_dir = Path(output_dir) if output_dir else PAPER_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"zones_avg_{suffix}.pdf"
    plt.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 2 saved to: {out}")


if __name__ == "__main__":
    main()
