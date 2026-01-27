import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rlg import PAPER_DIR, RUN_DIR
from rlg.plotting.utils import ANTHROPIC, apply_anthropic_style, setup_latex_fonts

sns.set_theme(style="darkgrid")


def main():
    setup_latex_fonts()

    left_policy_guarantees = RUN_DIR / "bridge_world/eval/left_bridge/guarantees.csv"
    left_policy_empirical = (
        RUN_DIR / "bridge_world/eval/left_bridge/empirical_safety.csv"
    )
    right_policy_guarantees = RUN_DIR / "bridge_world/eval/right_bridge/guarantees.csv"
    right_policy_empirical = (
        RUN_DIR / "bridge_world/eval/right_bridge/empirical_safety.csv"
    )

    left_policy_guarantees = pd.read_csv(left_policy_guarantees)
    left_policy_empirical = pd.read_csv(left_policy_empirical)
    right_policy_guarantees = pd.read_csv(right_policy_guarantees)
    right_policy_empirical = pd.read_csv(right_policy_empirical)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor("white")
    apply_anthropic_style(ax)

    # swapped: x = guarantees (t), y = probs (1-δ)
    ax.plot(
        left_policy_guarantees["guarantee"],
        left_policy_guarantees["prob"],
        color=ANTHROPIC["red_dark"],
        linewidth=2.8,
        solid_capstyle="round",
        drawstyle="steps-pre",
        label=r"\textbf{Certified Bound $1-\varepsilon$}",
    )
    ax.plot(
        left_policy_empirical["guarantee"],
        left_policy_empirical["empirical_safety"],
        color=ANTHROPIC["red_dark"],
        linewidth=2.8,
        linestyle="--",
        dash_capstyle="round",
        label=r"\textbf{Actual Safety $S_{\mathcal{D}}^\pi(B)$}",
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

    ax.set_xlim(left=0.5)
    fig.tight_layout(pad=0.8)
    out = PAPER_DIR / "plots/bridge_world.pdf"
    plt.savefig(out, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot 2 saved to: {out}")


if __name__ == "__main__":
    main()
