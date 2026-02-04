import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from rlg import PAPER_DIR, RUN_DIR
from rlg.plotting.utils import ANTHROPIC, apply_anthropic_style, setup_latex_fonts

sns.set_theme(style="darkgrid")


def main():
    setup_latex_fonts()

    dfs = {}
    for env in ["cheetah", "walker", "ZoneEnv"]:
        dfs[env] = load_returns(env, "main")
    dfs["bridge_left"] = load_returns("bridge_world", "left_bridge")
    dfs["bridge_right"] = load_returns("bridge_world", "right_bridge")

    out_dir = PAPER_DIR / "plots/performance_dists"
    out_dir.mkdir(parents=True, exist_ok=True)

    for env, df in dfs.items():
        plot_performance_histogram(df, env, out_dir)
        plot_performance_ecdf(df, env, out_dir)


def load_returns(env: str, run: str) -> pl.DataFrame:
    path = RUN_DIR / f"{env}/eval/{run}/episode_returns.parquet"
    df = pl.read_parquet(path)
    return df.group_by("task_id").agg(
        pl.col("total_return").mean().alias("mean_return")
    )


def plot_performance_histogram(df: pl.DataFrame, env: str, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("white")
    apply_anthropic_style(ax)

    returns = df.select(pl.col("mean_return")).to_numpy().flatten()

    sns.histplot(
        returns,
        bins=30,
        stat="proportion",
        color=ANTHROPIC["red_dark"],
        edgecolor=ANTHROPIC["ink"],
        kde=True,
        linewidth=0.8,
        alpha=0.85,
        ax=ax,
    )

    ax.set_xlabel(
        r"\textbf{Mean Episode Return}",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )
    ax.set_ylabel(
        r"\textbf{Proportion}",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )

    fig.tight_layout(pad=0.8)

    out_path = out_dir / f"{env}_histogram.pdf"
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def plot_performance_ecdf(df: pl.DataFrame, env: str, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("white")
    apply_anthropic_style(ax)

    returns = df.select(pl.col("mean_return")).to_numpy().flatten()

    sns.ecdfplot(
        returns,
        color=ANTHROPIC["red_dark"],
        linewidth=2.8,
        ax=ax,
    )

    ax.set_xlabel(
        r"\textbf{Mean Episode Return}",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )
    ax.set_ylabel(
        r"\textbf{Cumulative Proportion}",
        fontsize=28,
        color=ANTHROPIC["ink"],
        labelpad=6,
    )

    fig.tight_layout(pad=0.8)

    out_path = out_dir / f"{env}_ecdf.pdf"
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
