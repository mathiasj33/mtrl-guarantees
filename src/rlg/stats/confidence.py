"""Utilities for computing confidence intervals."""

import numpy as np
from scipy.stats import beta


def clopper_pearson(k, n, gamma=0.05):
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
    lower = beta.ppf(gamma, k, n - k + 1)
    lower = np.nan_to_num(lower, nan=0.0)
    return lower
