from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from jaxltl.utils.plot_utils import smooth

sns.set_theme(style="darkgrid")


def load_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bin_size = 4096 * 16
    df["bin"] = df["timestep"] // bin_size

    avg_data = (
        df.groupby(["seed", "bin"])[["return", "length", "curriculum_stage"]]
        .mean()
        .reset_index()
    )

    avg_data["timestep"] = avg_data["bin"] * bin_size

    for seed in avg_data["seed"].unique():
        mask = avg_data["seed"] == seed
        avg_data.loc[mask, "smooth_return"] = smooth(
            avg_data.loc[mask, "return"], radius=10
        )
        avg_data.loc[mask, "smooth_length"] = smooth(
            avg_data.loc[mask, "length"], radius=10
        )

    avg_data["name"] = Path(path).parent.name
    return avg_data


dfs = [
    load_df(p)
    for p in [
        # "runs/ZoneEnv/fix/logs.csv",
        # "runs/ZoneEnv/default/logs.csv",
        # "runs/ZoneEnv/tmp/logs.csv",
        # "runs/ZoneEnv/full/logs.csv",
        # "runs/ZoneEnv/full2/logs.csv",
        # "runs/RGBZoneEnv/tmp/logs.csv",
        "runs/ZoneEnv/deepltl/logs.csv",
    ]
]
df = pd.concat(dfs, ignore_index=True)


fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.lineplot(
    data=df, x="timestep", y="smooth_return", hue="name", ax=axes[0], legend=False
)
axes[0].set_title("Average Return")
axes[0].set_ylabel("Smoothed Return")

sns.lineplot(
    data=df, x="timestep", y="smooth_length", hue="name", ax=axes[1], legend=False
)
axes[1].set_title("Average Episode Length")
axes[1].set_ylabel("Smoothed Length")

sns.lineplot(
    data=df,
    x="timestep",
    y="curriculum_stage",
    hue="name",
    ax=axes[2],
)
axes[2].set_title("Curriculum Stage")
axes[2].set_ylabel("Curriculum Stage")

# Move the legend outside the plot area
sns.move_legend(axes[2], "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
