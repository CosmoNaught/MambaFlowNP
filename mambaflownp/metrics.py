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


def c2st(samples_a, samples_b, n_folds: int = 5, seed: int = 42):
    """Classifier two-sample test accuracy, averaged over query locations. At each location a
    one-dimensional logistic classifier separates the two sample sets under ``n_folds``-fold
    cross-validation; 0.5 means the marginals cannot be told apart. samples: (S, N)."""
    samples_a = np.asarray(samples_a, dtype=np.float64)
    samples_b = np.asarray(samples_b, dtype=np.float64)
    return float(np.mean([_c2st_1d(samples_a[:, i], samples_b[:, i], n_folds, seed + i)
                          for i in range(samples_a.shape[1])]))


def _c2st_1d(a, b, n_folds, seed):
    rng = np.random.RandomState(seed)
    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    x = (x - x.mean()) / (x.std() + 1e-8)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    fold = len(x) // n_folds
    accuracies = []
    for k in range(n_folds):
        test = slice(k * fold, (k + 1) * fold)
        train = np.ones(len(x), bool)
        train[test] = False
        w, c = 0.0, 0.0
        for _ in range(200):
            p = 1.0 / (1.0 + np.exp(-np.clip(x[train] * w + c, -30.0, 30.0)))
            w -= 0.1 * np.mean((p - y[train]) * x[train])
            c -= 0.1 * np.mean(p - y[train])
        accuracies.append(float(np.mean(((x[test] * w + c) > 0.0) == y[test])))
    return float(np.mean(accuracies))
