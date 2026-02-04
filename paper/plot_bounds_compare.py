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
    resolve_plot_combos,
    setup_latex_fonts,
    summarize_step_curves,
)

sns.set_theme(style="darkgrid")


def parse_bounds(text: str) -> list[str]:
    bounds = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not bounds:
        raise ValueError("No bounds parsed.")
    return bounds


def label_from_value(value: str) -> str:
    return value.replace("-", "m")


def env_xlim_left(env_name: str) -> float:
    if env_name.lower() == "zoneenv":
        return 0.0
    if env_name.lower() in {"walker", "cheetah"}:
        return 0.4
    return 0.0


def bound_label(bound: str) -> str:
    return bound.replace("-", " ").title()


def main() -> None:
    setup_latex_fonts()

    env = os.getenv("PLOT_ENV", "ZoneEnv")
    run = os.getenv("PLOT_RUN", "main")
    beta = os.getenv("PLOT_BETA", "1e-2")
    delta = os.getenv("PLOT_DELTA", "1e-4")
    bounds = parse_bounds(os.getenv("PLOT_BOUNDS", "hoeffding,dkw,bernstein"))

    combos = resolve_plot_combos(100, 1000)
    run_root = RUN_DIR / f"{env}/eval/{run}"
    empirical = pd.read_csv(run_root / "empirical_safety.csv")

    beta_label = label_from_value(beta)
    delta_label = label_from_value(delta)

    colors = plt.cm.tab10(np.linspace(0.1, 0.7, len(bounds)))
    x_left = env_xlim_left(env)

    output_dir = os.getenv("PLOT_OUTPUT_DIR")
    out_dir = Path(output_dir) if output_dir else PAPER_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for num_tasks, num_episodes in combos:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        fig.patch.set_facecolor("white")
        apply_anthropic_style(ax)

        for bound, color in zip(bounds, colors):
            label = f"{bound}_beta{beta_label}_delta{delta_label}".replace("-", "m")
            bound_dir = run_root / "ablations" / label
            guarantees = pd.read_csv(
                bound_dir / guarantees_batches_filename(num_tasks, num_episodes)
            )
            grid, mean, std = summarize_step_curves(
                guarantees,
                num_tasks=num_tasks,
                num_episodes=num_episodes,
                num_points=600,
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
                label=rf"\textbf{{{bound_label(bound)}}}",
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

        ax.set_xlim(left=x_left)
        fig.tight_layout(pad=0.8)
        out = out_dir / f"bounds_compare_tasks{num_tasks}_episodes{num_episodes}.pdf"
        plt.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {out}")


if __name__ == "__main__":
    main()
