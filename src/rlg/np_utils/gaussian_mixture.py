import numpy as np
from scipy.stats import norm


class GaussianMixture:
    """
    A class to represent a 1D Mixture of Gaussians.

    Attributes:
        weights (np.ndarray): The mixing coefficients (must sum to 1).
        means (np.ndarray): The means of the individual Gaussian components.
        stds (np.ndarray): The standard deviations of the individual Gaussian components.
        n_components (int): The number of Gaussian components.
    """

    def __init__(self, weights, means, stds):
        """
        Initialize the Gaussian Mixture.

        Args:
            weights (list or np.ndarray): Mixing weights for each component.
                                          Will be normalized to sum to 1.
            means (list or np.ndarray): Means (mu) for each component.
            stds (list or np.ndarray): Standard deviations (sigma) for each component.
        """
        self.weights = np.array(weights, dtype=float)
        self.means = np.array(means, dtype=float)
        self.stds = np.array(stds, dtype=float)

        # Validation
        if not (len(self.weights) == len(self.means) == len(self.stds)):
            raise ValueError(
                "Input arrays (weights, means, stds) must have the same length."
            )

        self.n_components = len(self.weights)

        # Normalize weights to ensure they sum to 1.0
        if not np.isclose(self.weights.sum(), 1.0):
            print(
                f"Warning: Weights summed to {self.weights.sum()}. Normalizing to 1.0."
            )
            self.weights /= self.weights.sum()

    def sample(self):
        """
        Sample a value from the Gaussian Mixture.
        """
        component = np.random.choice(self.n_components, p=self.weights)
        sample = np.random.normal(loc=self.means[component], scale=self.stds[component])
        return sample

    def pdf(self, x):
        """
        Calculate the Probability Density Function (PDF) of the mixture.

        PDF(x) = sum( weight_i * Normal(x | mu_i, sigma_i) )
        """
        # Initialize result array
        result = np.zeros_like(x, dtype=float)

        for w, mu, sigma in zip(self.weights, self.means, self.stds, strict=False):
            result += w * norm.pdf(x, loc=mu, scale=sigma)

        return result

    def cdf(self, x):
        """
        Calculate the Cumulative Distribution Function (CDF) of the mixture.

        CDF(x) = sum( weight_i * CDF_Normal(x | mu_i, sigma_i) )
        """
        # Initialize result array
        result = np.zeros_like(x, dtype=float)

        for w, mu, sigma in zip(self.weights, self.means, self.stds, strict=False):
            result += w * norm.cdf(x, loc=mu, scale=sigma)

        return result
