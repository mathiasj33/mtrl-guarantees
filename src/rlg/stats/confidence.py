"""Utilities for computing confidence intervals."""

import numpy as np
import polars as pl
from scipy import stats


def clopper_pearson(k, n, beta=0.05):
    """
    Computes Clopper-Pearson confidence intervals for binomial success.

    Args:
        k: Number of successes (int or array-like)
        n: Number of trials (int or array-like)
        gamma: Significance level (float), default 0.05

    Returns:
        lower_bound: Lower bound of the confidence interval (float or array-like)
    """
    # Ensure inputs are numpy arrays for vectorized operations
    k = np.array(k)
    n = np.array(n)

    # --- Lower bound ---
    # Use beta.ppf(alpha, k, n - k + 1)
    # The ppf (Percent Point Function) is the inverse of the CDF.
    # We use np.nan_to_num to handle the edge case k=0, which returns NaN
    lower = stats.beta.ppf(beta, k, n - k + 1)
    lower = np.nan_to_num(lower, nan=0.0)
    return lower


def hoeffding(df, min_return, max_return, beta=0.05):
    """
    Computes Hoeffding confidence intervals for real-valued returns.

    Args:
        df: Polars dataframe with task_id, episode_id, total_return columns
        min_return: Minimum possible return (float)
        max_return: Maximum possible return (float)
        gamma: Significance level (float), default 0.05
    Returns:
        lower_bound: dataframe with task_id, mean_return, and lower_bound columns
    """

    episodes_per_task = df.group_by("task_id").agg(
        pl.col("episode_id").count().alias("n_episodes")
    )
    assert episodes_per_task.select(pl.col("n_episodes").n_unique()).item() == 1, (
        "All tasks must have the same number of episodes"
    )
    n_episodes = episodes_per_task.select(pl.col("n_episodes").first()).item()

    # Compute mean return per task
    mean_returns = df.group_by("task_id").agg(
        pl.col("total_return").mean().alias("mean_return")
    )

    # Hoeffding lower bound calculation
    hoeffding_bound = (max_return - min_return) * np.sqrt(
        (np.log(1 / beta)) / (2 * n_episodes)
    )

    # Compute lower bounds
    mean_returns = mean_returns.with_columns(
        (pl.col("mean_return") - hoeffding_bound).alias("lower_bound")
    )

    return mean_returns.select(["task_id", "mean_return", "lower_bound"]).sort(
        "task_id"
    )


def empirical_bernstein(df, min_return, max_return, beta=0.05):
    """
    Computes Empirical Bernstein lower bounds for real-valued returns.

    The bound is based on Maurer and Pontil (2009). It adapts to the
    sample variance, often providing tighter bounds than Hoeffding when
    variance is low.

    Args:
        df: Polars dataframe with task_id, episode_id, total_return columns
        min_return: Minimum possible return (float)
        max_return: Maximum possible return (float)
        gamma: Significance level (float), default 0.05
    Returns:
        dataframe with task_id, mean_return, and lower_bound columns
    """

    # 1. Validate sample size consistency
    episodes_per_task = df.group_by("task_id").agg(
        pl.col("episode_id").count().alias("n_episodes")
    )

    assert episodes_per_task.select(pl.col("n_episodes").n_unique()).item() == 1, (
        "All tasks must have the same number of episodes"
    )

    n = episodes_per_task.select(pl.col("n_episodes").first()).item()

    if n < 2:
        raise ValueError(
            "Empirical Bernstein requires at least 2 samples to compute variance."
        )

    # 2. Compute Mean and Unbiased Variance per task
    stats = df.group_by("task_id").agg(
        [
            pl.col("total_return").mean().alias("mean_return"),
            # Polars .var() computes unbiased sample variance (ddof=1) by default
            pl.col("total_return").var().alias("sample_var"),
        ]
    )

    # 3. Scale statistics to the [0, 1] domain
    # The theorem is strictly defined for random variables in [0, 1].
    # We must normalize the variance: Var(X_scaled) = Var(X) / (range)^2
    range_span = max_return - min_return

    stats = stats.with_columns(
        (pl.col("sample_var") / (range_span**2)).alias("scaled_var")
    )

    # 4. Calculate the Deviation Term (epsilon)
    # Formula: sqrt(2 * Var * ln(2/d) / n) + 7 * ln(2/d) / (3 * (n - 1))
    # Note: We use ln(2/gamma) because the empirical bound requires a union bound
    # (cost of estimating both mean and variance).
    log_term = np.log(2 / beta)

    stats = stats.with_columns(
        (
            np.sqrt(2 * pl.col("scaled_var") * log_term / n)
            + (7 * log_term) / (3 * (n - 1))
        ).alias("epsilon_scaled")
    )

    # 5. Compute Lower Bound and Denormalize
    # Bound = Mean - (epsilon_scaled * range_span)
    stats = stats.with_columns(
        (pl.col("mean_return") - (pl.col("epsilon_scaled") * range_span)).alias(
            "lower_bound"
        )
    )

    return stats.select(["task_id", "mean_return", "lower_bound"]).sort("task_id")
