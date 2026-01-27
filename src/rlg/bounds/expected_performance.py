"""Utilities for computing bounds on expected performance."""

from typing import cast

from joblib import Parallel, delayed
from scipy.optimize import bisect
from scipy.stats import binom


def compute_guarantees(
    lower_bounds: list[float],
    min_return: float,
    max_return: float,
    beta: float,
    delta: float,
    step_size: int = 5,
    n_jobs: int = 8,
) -> tuple[list[float], list[float]]:
    """
    Computes performance guarantees given lower bounds, maximum discard fraction,
    gamma, and eta.

    Parameters:
        lower_bounds (list[float]): List of lower bounds on performance metric.
        max_discard_frac (float): Maximum fraction of samples that can be discarded.
        beta (float): Failure probability of individual verification.
        delta (float): Desired reliability level (1 - delta).
        step_size (int): Step size for iterating over lower bounds.
        n_jobs (int): Number of parallel jobs to use.

    Returns:
        tuple of lists of performance guarantees and probabilities with which they are
        satisfied.
    """
    lower_bounds = sorted(lower_bounds)
    max_k = int(0.99 * len(lower_bounds))
    k_values = list(range(0, max_k, step_size))
    k_values.append(max_k)

    probs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_optimal_prob_with_discard)(len(lower_bounds), k, beta, delta)
        for k in k_values
    )
    probs = cast(list[float], probs)
    guarantees = [lower_bounds[k] for k in k_values]
    guarantees = [min_return] + guarantees + [max_return]
    probs = [1.0] + probs + [0.0]
    return guarantees, probs


def compute_optimal_prob_with_discard(
    n: int, max_discard: int, gamma: float, eta: float
) -> float:
    """
    Computes the optimal success probability (1-epsilon) with discarded samples.

    Parameters:
        n (int): Total number of samples.
        max_discard (int): Maximum number of discarded samples.
        gamma (float): Failure probability of individual verification.
        eta (float): Desired reliability level (1 - eta).

    Returns:
        float: Optimal success probability (1-epsilon).
    """
    min_epsilon = 1

    for k in range(max(1, int((n - max_discard) * 0.7)), n + 1 - max_discard):
        p = 1 - binom.cdf(k, n - max_discard, 1 - gamma)
        if p < 1 - eta:
            continue

        beta = p - (1 - eta)
        epsilon = bisect(binom_cdf_diff, 0, 1, args=(n, n - k, beta))

        min_epsilon = min(min_epsilon, epsilon)

    return 1 - min_epsilon  # type: ignore


def binom_cdf_diff(p, n, k, alpha):
    """
    Computes the difference between the binomial CDF and the given threshold alpha.

    Parameters:
        p (float): Probability of success.
        n (int): Number of trials.
        k (int): Number of successes.
        alpha (float): Threshold value.

    Returns:
        float: Difference between the CDF and alpha.
    """
    return binom.cdf(k, n, p) - alpha
