# --- Warm "Anthropic-ish" palette (high contrast pair) ---
from matplotlib import pyplot as plt

ANTHROPIC = {
    "red_dark": "#9A3412",  # deep warm red-brown
    "orange_light": "#FDBA74",  # light peach/orange
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


def apply_anthropic_style(ax: plt.Axes) -> None:
    ax.set_facecolor((0.98, 0.95, 0.95, 0.3))
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
        ax.spines[side].set_linewidth(3.2)

    # ticks
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=24,
        width=1.9,
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
