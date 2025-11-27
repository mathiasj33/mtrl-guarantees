"""Utility functions for plotting."""

import numpy as np


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, "same") / np.convolve(z, y, "same")
