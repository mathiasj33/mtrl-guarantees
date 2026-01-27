import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rlg import PAPER_DIR, RUN_DIR
from rlg.plotting.utils import ANTHROPIC, apply_anthropic_style, setup_latex_fonts

sns.set_theme(style="darkgrid")


def main():
    setup_latex_fonts()

    guarantees = pd.read_csv(RUN_DIR / "ZoneEnv/eval/main/guarantees.csv")
    empirical = pd.read_csv(RUN_DIR / "ZoneEnv/eval/main/empirical_safety.csv")
    guarantees = guarantees.query("num_tasks == 200 & num_episodes == 1000")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("white")
    apply_anthropic_style(ax)

    # swapped: x = guarantees (t), y = probs (1-δ)
    ax.plot(
        guarantees["guarantee"],
        guarantees["prob"],
        color=ANTHROPIC["red_dark"],
        linewidth=2.8,
        solid_capstyle="round",
        drawstyle="steps-pre",
        label=r"\textbf{Certified Bound}",
    )
    ax.plot(
        empirical["guarantee"],
        empirical["empirical_safety"],
        color=ANTHROPIC["red_dark_translucent"],
        linewidth=2.8,
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

    ax.set_xlim(left=0.4)
    fig.tight_layout(pad=0.8)
    out = PAPER_DIR / "plots/zones.pdf"
    plt.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 2 saved to: {out}")


if __name__ == "__main__":
    main()
