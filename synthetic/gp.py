"""Synthetic Gaussian-process tasks (TNP settings, Nguyen and Grover, 2022) and their model,
training and evaluation settings.

Each task draws a fresh GP per episode with lengthscale ~ U(0.1, 0.6), output scale ~ U(0.1, 1.0)
and noise standard deviation 0.02. 1D inputs lie in [-2, 2] and 2D inputs in [0, 1]^2; the model
sees coordinates rescaled to the unit cube.
"""

import dataclasses
import math
import pathlib
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mambaflownp.model import Config, MambaFlowNP, load_checkpoint

NOISE_STD = 0.02
LENGTH_RANGE = (0.1, 0.6)
SCALE_RANGE = (0.1, 1.0)

TASKS = {
    "1d_rbf": dict(kernel="rbf", coord_dim=1, domain=(-2.0, 2.0)),
    "1d_matern32": dict(kernel="matern32", coord_dim=1, domain=(-2.0, 2.0)),
    "1d_matern52": dict(kernel="matern52", coord_dim=1, domain=(-2.0, 2.0)),
    "2d_matern32": dict(kernel="matern32", coord_dim=2, domain=(0.0, 1.0)),
}

_POINTS = {
    "1d_rbf": dict(max_points=128, coord_dim=1),
    "1d_matern32": dict(max_points=128, coord_dim=1),
    "1d_matern52": dict(max_points=128, coord_dim=1),
    "2d_matern32": dict(max_points=256, coord_dim=2),
}

# Hyperparameters selected by Bayesian optimisation on validation MSE.
TUNED = {
    "1d_rbf": dict(hidden_size=384, d_state=32, num_layers=3,
                   rff_freqs=128, c_scale=0.5, lr=0.000843, batch_size=128),
    "1d_matern32": dict(hidden_size=128, d_state=64, num_layers=8,
                        rff_freqs=192, c_scale=0.5, lr=0.000994, batch_size=128),
    "1d_matern52": dict(hidden_size=128, d_state=16, num_layers=3,
                        rff_freqs=192, c_scale=1.0, lr=0.00163, batch_size=128),
    "2d_matern32": dict(hidden_size=256, d_state=128, num_layers=6,
                        rff_freqs=384, c_scale=0.25, lr=0.00111, batch_size=64),
}

_ARCH = ("hidden_size", "d_state", "num_layers", "rff_freqs", "c_scale")
MODELS = {k: Config(**{a: v[a] for a in _ARCH}, **_POINTS[k]) for k, v in TUNED.items()}

_TRAIN = dict(num_steps=80_000, distill_steps=20_000, n_couplings=2_000, coupling_ode_steps=10,
              min_ctx=16, min_qry=16)
TRAIN = {k: dict(lr=v["lr"], batch_size=v["batch_size"], **_TRAIN) for k, v in TUNED.items()}

EVAL = {
    "1d_rbf": dict(num_draws=128),
    "1d_matern32": dict(num_draws=128),
    "1d_matern52": dict(num_draws=128),
    "2d_matern32": dict(num_draws=128),
}

VARIANTS = {
    "full": {},
    "single_direction": dict(num_scan_dirs=1),
    "two_directions": dict(num_scan_dirs=2),
    "random_order": dict(ordering="random"),
    "single_hidden_state": dict(ordering="context_first", num_scan_dirs=1),
    "morton": dict(ordering="morton"),
    "lexicographic": dict(ordering="lexicographic"),
}

ROOT = pathlib.Path(__file__).resolve().parents[1]


def data_path(name, seed):
    return ROOT / "data" / "synthetic" / f"{name}_seed{seed}.npz"


def config(name, variant="full"):
    """Model configuration of a task, with an ablation ``variant`` applied."""
    return dataclasses.replace(MODELS[name], **VARIANTS[variant])


def load_model(name, weights, variant="full"):
    """The model of a synthetic task, loaded from a checkpoint written by synthetic/train.py."""
    model = MambaFlowNP(config(name, variant), rngs=nnx.Rngs(0))
    return load_checkpoint(model, weights)


def to_unit(name, x):
    """Rescales task coordinates to the unit cube seen by the model."""
    lo, hi = TASKS[name]["domain"]
    return ((np.asarray(x, dtype=np.float32) - lo) / (hi - lo)).astype(np.float32)


def covariance(x, kernel, length, scale, noise_std, xp=np):
    """Kernel matrix of the points x (n, d) plus observation noise."""
    diff = (x[:, None, :] - x[None, :, :]) / length
    sqdist = xp.sum(diff ** 2, axis=-1)
    dist = xp.sqrt(xp.maximum(sqdist, 0.0))
    if kernel == "rbf":
        k = xp.exp(-0.5 * sqdist)
    elif kernel == "matern32":
        k = (1.0 + math.sqrt(3.0) * dist) * xp.exp(-math.sqrt(3.0) * dist)
    elif kernel == "matern52":
        k = (1.0 + math.sqrt(5.0) * dist + 5.0 * sqdist / 3.0) * xp.exp(-math.sqrt(5.0) * dist)
    else:
        raise ValueError(f"unknown kernel {kernel}")
    return scale ** 2 * k + noise_std ** 2 * xp.eye(x.shape[0], dtype=k.dtype)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6))
def _draw(key, kernel, batch_size, num_points, coord_dim, lo, hi):
    key_x, key_l, key_s, key_y = jax.random.split(key, 4)
    x = jax.random.uniform(key_x, (batch_size, num_points, coord_dim), minval=lo, maxval=hi)
    length = jax.random.uniform(key_l, (batch_size,), minval=LENGTH_RANGE[0],
                                maxval=LENGTH_RANGE[1])
    scale = jax.random.uniform(key_s, (batch_size,), minval=SCALE_RANGE[0],
                               maxval=SCALE_RANGE[1])
    cov = jax.vmap(lambda xi, li, si: covariance(xi, kernel, li, si, NOISE_STD, xp=jnp))(
        x, length, scale)
    chol = jnp.linalg.cholesky(cov + 1e-6 * jnp.eye(num_points))
    y = jnp.einsum("bij,bj->bi", chol, jax.random.normal(key_y, (batch_size, num_points)))
    return x, y[..., None]


class TaskSampler:
    """Samples padded training episodes of one synthetic task.

    Each batch shares one split: N ~ U{min_ctx, K - min_qry} context points and
    M ~ U{min_qry, K - N} query points, with K = max_points.
    """

    def __init__(self, name, seed=None):
        self.name, self.task = name, TASKS[name]
        self.max_points = MODELS[name].max_points
        self.rng = np.random.default_rng(seed)
        self.key = jax.random.PRNGKey(0 if seed is None else seed)

    def sample_padded(self, batch_size: int, min_ctx: int = 16, min_qry: int = 16):
        """Returns padded (x_ctx, y_ctx, x_qry, y_qry, ctx_mask, qry_mask) in model coordinates."""
        k = self.max_points
        ctx_pad, qry_pad = k - min_qry, k - min_ctx
        n_ctx = int(self.rng.integers(min_ctx, ctx_pad + 1))
        n_qry = int(self.rng.integers(min_qry, k - n_ctx + 1))
        self.key, draw_key = jax.random.split(self.key)
        lo, hi = self.task["domain"]
        x, y = _draw(draw_key, self.task["kernel"], batch_size, k, self.task["coord_dim"],
                     lo, hi)
        x = (x - lo) / (hi - lo)
        qry_idx = jnp.minimum(n_ctx + jnp.arange(qry_pad), k - 1)
        ctx_mask = jnp.broadcast_to(jnp.arange(ctx_pad) < n_ctx, (batch_size, ctx_pad))
        qry_mask = jnp.broadcast_to(jnp.arange(qry_pad) < n_qry, (batch_size, qry_pad))
        return (x[:, :ctx_pad] * ctx_mask[:, :, None], y[:, :ctx_pad] * ctx_mask[:, :, None],
                jnp.take(x, qry_idx, axis=1) * qry_mask[:, :, None],
                jnp.take(y, qry_idx, axis=1) * qry_mask[:, :, None], ctx_mask, qry_mask)


def load_episodes(name, seed):
    """Cached test episodes of one evaluation seed, in task coordinates."""
    with np.load(data_path(name, seed)) as f:
        return {k: f[k] for k in f.files}


def gp_posterior_samples(x_ctx, y_ctx, x_qry, kernel, length, scale, num_draws, seed=0):
    """Exact GP posterior draws at x_qry given (x_ctx, y_ctx), shape (num_draws, n_qry)."""
    x = np.concatenate([x_ctx, x_qry]).astype(np.float64)
    n = len(x_ctx)
    cov = covariance(x, kernel, max(float(length), 1e-6), float(scale), NOISE_STD)
    cc, tc, tt = cov[:n, :n], cov[n:, :n], cov[n:, n:]
    jitter, eye_c, eye_t = 1e-6, np.eye(n), np.eye(len(x_qry))
    y = np.asarray(y_ctx, dtype=np.float64).reshape(-1)
    for _ in range(8):
        try:
            mean = tc @ np.linalg.solve(cc + jitter * eye_c, y)
            post = tt - tc @ np.linalg.solve(cc + jitter * eye_c, tc.T)
            post = 0.5 * (post + post.T)
            np.linalg.cholesky(post + jitter * eye_t)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        mean = tc @ np.linalg.solve(cc + jitter * eye_c, y)
        post = np.diag(np.maximum(np.diag(tt), 1e-6))
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, post + 1e-6 * eye_t, size=num_draws).astype(np.float32)
