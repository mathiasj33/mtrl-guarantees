# --- Warm "Anthropic-ish" palette (high contrast pair) ---
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

COLORS = {
    "red_dark": "#9A3412",  # deep warm red-brown
    "red_dark_translucent": (154 / 255, 52 / 255, 18 / 255, 0.5),
    "orange_light": "#FDBA74",  # light peach/orange
    "orange_light_translucent": (253 / 255, 186 / 255, 116 / 255, 0.5),
    "ink": "#0f172a",  # slate/near-black
    "grid": "#e5e7eb",  # very light gray
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


def apply_style(ax: plt.Axes) -> None:
    ax.set_facecolor((0.98, 0.95, 0.95, 0.3))
    ax.set_axisbelow(True)

    # subtle dashed major grid only
    ax.grid(
        True,
        which="major",
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        color=COLORS["grid"],
        alpha=0.9,
    )
    ax.grid(False, which="minor")

    # show all spines, thicker
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(COLORS["ink"])
        ax.spines[side].set_linewidth(3.2)

    # ticks
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
        width=1.9,
        length=6,
        colors=COLORS["ink"],
        pad=3,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.2,
        length=3,
        colors=COLORS["ink"],
    )


def summarize_step_curves(
    df: pd.DataFrame,
    num_tasks: int,
    num_episodes: int,
    num_points: int = 500,
    batch_col: str = "batch_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = df.query("num_tasks == @num_tasks and num_episodes == @num_episodes")
    if subset.empty:
        raise ValueError(
            "No guarantees found for the requested num_tasks/num_episodes combination."
        )

    min_g = subset["guarantee"].min()
    max_g = subset["guarantee"].max()
    grid = np.linspace(min_g, max_g, num_points)

    curves = []
    for _, group in subset.groupby(batch_col):
        guarantees = group["guarantee"].to_numpy()
        probs = group["prob"].to_numpy()
        order = np.argsort(guarantees)
        guarantees = guarantees[order]
        probs = probs[order]
        idx = np.searchsorted(guarantees, grid, side="right")
        idx = np.clip(idx, 0, len(probs) - 1)
        curves.append(probs[idx])

    stacked = np.vstack(curves)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return grid, mean, std


def resolve_plot_params(default_tasks: int, default_episodes: int) -> tuple[int, int]:
    tasks = int(os.getenv("PLOT_NUM_TASKS", default_tasks))
    episodes = int(os.getenv("PLOT_NUM_EPISODES", default_episodes))
    return tasks, episodes


def get_guarantees_filename(num_tasks: int, num_episodes: int) -> str:
    return f"guarantees_tasks{num_tasks}_episodes{num_episodes}.csv"


def resolve_plot_combos(combinations: list[str]) -> list[tuple[int, int]]:
    combos = []
    for chunk in combinations:
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" in chunk:
            left, right = chunk.split("x", 1)
        elif ":" in chunk:
            left, right = chunk.split(":", 1)
        else:
            raise ValueError(
                "PLOT_COMBOS entries must be formatted like 'tasksxepisodes'."
            )
        combos.append((int(left), int(right)))
    return combos


def plot_suffix_from_combos(combos: list[tuple[int, int]]) -> str:
    parts = [f"tasks{tasks}_episodes{episodes}" for tasks, episodes in combos]
    return "_".join(parts)
