import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rlg import RUN_DIR

sns.set_theme(style="darkgrid")


def main():
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

    colors = sns.color_palette()

    plt.plot(
        left_policy_guarantees["guarantee"],
        left_policy_guarantees["prob"],
        label="Left (Bound)",
        color=colors[0],
    )
    plt.plot(
        left_policy_empirical["guarantee"],
        left_policy_empirical["empirical_safety"],
        label="Left (Empirical)",
        linestyle="--",
        color=colors[0],
    )

    plt.plot(
        right_policy_guarantees["guarantee"],
        right_policy_guarantees["prob"],
        label="Right (Bound)",
        color=colors[1],
    )
    plt.plot(
        right_policy_empirical["guarantee"],
        right_policy_empirical["empirical_safety"],
        label="Right (Empirical)",
        linestyle="--",
        color=colors[1],
    )

    plt.xlabel("Guarantee")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
