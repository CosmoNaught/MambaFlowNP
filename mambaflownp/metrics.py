"""Probabilistic evaluation metrics."""

import numpy as np


def crps(samples, truth):
    """Mean CRPS over points. samples: (S, N); truth: (N,)."""
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    abs_diff = np.mean(np.abs(samples - truth[None, :]), axis=0)
    n = len(samples)
    w = 2 * np.arange(1, n + 1) - n - 1
    spread = np.mean(np.sort(samples, axis=0) * w[:, None], axis=0) / n
    return float(np.mean(abs_diff - spread))


def interval_coverage(samples, truth, level):
    """Fraction of points inside the central sample interval at ``level``."""
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    alpha = (1.0 - level) / 2.0
    lo = np.quantile(samples, alpha, axis=0)
    hi = np.quantile(samples, 1.0 - alpha, axis=0)
    return float(np.mean((truth >= lo) & (truth <= hi)))
