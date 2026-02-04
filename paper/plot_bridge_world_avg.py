import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rlg import PAPER_DIR, RUN_DIR
from rlg.plotting.utils import (
    ANTHROPIC,
    apply_anthropic_style,
    combo_color,
    guarantees_batches_filename,
    plot_suffix_from_combos,
    resolve_plot_combos,
    setup_latex_fonts,
    summarize_step_curves,
)

sns.set_theme(style="darkgrid")


def main():
    setup_latex_fonts()

    left_empirical = pd.read_csv(
        RUN_DIR / "bridge_world/eval/left_bridge/empirical_safety.csv"
    )
    right_empirical = pd.read_csv(
        RUN_DIR / "bridge_world/eval/right_bridge/empirical_safety.csv"
    )
    combos = resolve_plot_combos(100, 1000)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("white")
    apply_anthropic_style(ax)

    for num_tasks, num_episodes in combos:
        left_color = combo_color(num_tasks, num_episodes, "Reds")
        right_color = combo_color(
            num_tasks, num_episodes, "YlOrBr", low=0.325, high=0.675
        )
        left_guarantees = pd.read_csv(
            RUN_DIR
            / "bridge_world/eval/left_bridge"
            / guarantees_batches_filename(num_tasks, num_episodes)
        )
        right_guarantees = pd.read_csv(
            RUN_DIR
            / "bridge_world/eval/right_bridge"
            / guarantees_batches_filename(num_tasks, num_episodes)
        )
        left_grid, left_mean, left_std = summarize_step_curves(
            left_guarantees,
            num_tasks=num_tasks,
            num_episodes=num_episodes,
            num_points=600,
        )
        right_grid, right_mean, right_std = summarize_step_curves(
            right_guarantees,
            num_tasks=num_tasks,
            num_episodes=num_episodes,
            num_points=600,
        )

        left_low = np.clip(left_mean - 1.5 * left_std, 0.0, 1.0)
        left_high = np.clip(left_mean + 1.5 * left_std, 0.0, 1.0)
        ax.fill_between(
            left_grid,
            left_low,
            left_high,
            color=left_color,
            alpha=0.22,
            step="pre",
        )
        ax.plot(
            left_grid,
            left_mean,
            color=left_color,
            linewidth=2.8,
            solid_capstyle="round",
            drawstyle="steps-pre",
            label=rf"\textbf{{Left Bridge ({num_tasks}, {num_episodes})}}",
        )

        right_low = np.clip(right_mean - 1.5 * right_std, 0.0, 1.0)
        right_high = np.clip(right_mean + 1.5 * right_std, 0.0, 1.0)
        ax.fill_between(
            right_grid,
            right_low,
            right_high,
            color=right_color,
            alpha=0.22,
            step="pre",
        )
        ax.plot(
            right_grid,
            right_mean,
            color=right_color,
            linewidth=2.8,
            solid_capstyle="round",
            drawstyle="steps-pre",
            label=rf"\textbf{{Right Bridge ({num_tasks}, {num_episodes})}}",
        )

    ax.plot(
        left_empirical["guarantee"],
        left_empirical["empirical_safety"],
        color=combo_color(combos[0][0], combos[0][1], "Reds"),
        linewidth=3.2,
        linestyle="--",
        dash_capstyle="round",
    )
    ax.plot(
        right_empirical["guarantee"],
        right_empirical["empirical_safety"],
        color=combo_color(
            combos[0][0], combos[0][1], "YlOrBr", low=0.325, high=0.675
        ),
        linewidth=3.2,
        linestyle="--",
        dash_capstyle="round",
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
    out = PAPER_DIR / f"plots/bridge_world_avg_{suffix}.pdf"
    plt.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 2 saved to: {out}")


if __name__ == "__main__":
    main()
