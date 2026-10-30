"""Coordinate projection, value normalization and task sampling."""

import math

import jax
import jax.numpy as jnp
import numpy as np


class CoordFrame:
    """Equirectangular projection of (longitude, latitude) onto the unit square."""

    def __init__(self, lat0, lo, hi):
        self.lat0 = float(lat0)
        self.lo = np.asarray(lo, dtype=np.float32)
        self.hi = np.asarray(hi, dtype=np.float32)

    @staticmethod
    def _project(lonlat, lat0):
        lonlat = np.asarray(lonlat, dtype=np.float32)
        x_km = np.deg2rad(lonlat[..., 0]) * np.float32(6371.0 * math.cos(lat0))
        y_km = np.deg2rad(lonlat[..., 1]) * np.float32(6371.0)
        return np.stack([x_km, y_km], axis=-1)

    @classmethod
    def fit(cls, lonlat):
        lat0 = np.deg2rad(float(np.mean(np.asarray(lonlat, dtype=np.float32)[:, 1])))
        km = cls._project(lonlat, lat0)
        return cls(lat0, km.min(axis=0), km.max(axis=0))

    def transform(self, lonlat):
        km = self._project(lonlat, self.lat0)
        span = self.hi - self.lo
        return np.where(span < 1e-6, 0.0, (km - self.lo) / np.maximum(span, 1e-6)).astype(
            np.float32)

    def inverse(self, coords):
        km = np.asarray(coords, dtype=np.float32) * (self.hi - self.lo) + self.lo
        lon = np.rad2deg(km[..., 0] / (6371.0 * math.cos(self.lat0)))
        lat = np.rad2deg(km[..., 1] / 6371.0)
        return np.stack([lon, lat], axis=-1).astype(np.float32)


class QuantileNormalizer:
    """Empirical quantile transform to a standard normal."""

    def __init__(self, raw_knots, z_knots, z_unique, raw_unique):
        self.raw_knots = np.asarray(raw_knots, dtype=np.float32)
        self.z_knots = np.asarray(z_knots, dtype=np.float32)
        self.z_unique = np.asarray(z_unique, dtype=np.float32)
        self.raw_unique = np.asarray(raw_unique, dtype=np.float32)

    @classmethod
    def fit(cls, values):
        values = np.asarray(values, dtype=np.float32)
        raw = np.sort(values[np.isfinite(values)])
        probs = np.clip((np.arange(raw.size, dtype=np.float32) + 0.5) / raw.size, 1e-6, 1 - 1e-6)
        z = np.sqrt(2.0) * np.asarray(jax.scipy.special.erfinv(jnp.asarray(2.0 * probs - 1.0)))
        raw_knots, starts, counts = np.unique(raw, return_index=True, return_counts=True)
        z_knots = np.add.reduceat(z, starts) / counts
        z_unique, starts_z, counts_z = np.unique(z, return_index=True, return_counts=True)
        raw_unique = np.add.reduceat(raw, starts_z) / counts_z
        return cls(raw_knots, z_knots, z_unique, raw_unique)

    @staticmethod
    def _interp(values, xp, fp):
        values = np.asarray(values, dtype=np.float32)
        out = np.full(values.shape, np.nan, dtype=np.float32)
        ok = np.isfinite(values)
        out[ok] = np.interp(values[ok], xp, fp, left=fp[0], right=fp[-1])
        return out

    def transform(self, values):
        return self._interp(values, self.raw_knots, self.z_knots)

    def inverse(self, values):
        return self._interp(values, self.z_unique, self.raw_unique)


class FieldSampler:
    """Samples context/query tasks from observations at fixed sites.

    Args:
      coords: Site coordinates in [0, 1], shape (num_sites, 2).
      values: Observations, shape (num_times, num_sites), NaN where missing.
      seed: Random seed.
    """

    def __init__(self, coords, values, seed=None):
        self.coords = np.asarray(coords, dtype=np.float32)
        self.values = np.asarray(values, dtype=np.float32)
        self.observed = np.isfinite(self.values)
        self.rng = np.random.default_rng(seed)

    def _eligible(self, num_points):
        eligible = np.flatnonzero(self.observed.sum(axis=1) >= num_points)
        if eligible.size == 0:
            raise RuntimeError(f"no time has {num_points} observed sites")
        return eligible

    def _draw(self, batch_size, num_points):
        eligible = self._eligible(num_points)
        x = np.empty((batch_size, num_points, 2), dtype=np.float32)
        y = np.empty((batch_size, num_points, 1), dtype=np.float32)
        for b in range(batch_size):
            when = int(self.rng.choice(eligible))
            chosen = self.rng.choice(np.flatnonzero(self.observed[when]), size=num_points,
                                     replace=False)
            chosen = chosen[self.rng.permutation(num_points)]
            x[b], y[b, :, 0] = self.coords[chosen], self.values[when, chosen]
        return x, y

    def sample(self, batch_size: int, n_ctx: int, n_qry: int):
        """Returns (x_ctx, y_ctx, x_qry, y_qry) with fixed context and query sizes."""
        x, y = self._draw(batch_size, n_ctx + n_qry)
        return x[:, :n_ctx], y[:, :n_ctx], x[:, n_ctx:], y[:, n_ctx:]

    def sample_padded(self, batch_size: int, max_points: int, min_ctx: int = 16,
                      min_qry: int = 16):
        """Returns padded (x_ctx, y_ctx, x_qry, y_qry, ctx_mask, qry_mask) with a random split."""
        max_points = int(min(max_points, self.coords.shape[0]))
        ctx_pad, qry_pad = max_points - min_qry, max_points - min_ctx
        n_ctx = int(self.rng.integers(min_ctx, ctx_pad + 1))
        x, y = self._draw(batch_size, max_points)

        qry_idx = np.minimum(n_ctx + np.arange(qry_pad), max_points - 1)
        ctx_mask = np.broadcast_to(np.arange(ctx_pad) < n_ctx, (batch_size, ctx_pad)).copy()
        qry_mask = np.broadcast_to(np.arange(qry_pad) < max_points - n_ctx,
                                   (batch_size, qry_pad)).copy()
        return (x[:, :ctx_pad] * ctx_mask[:, :, None], y[:, :ctx_pad] * ctx_mask[:, :, None],
                np.take(x, qry_idx, axis=1) * qry_mask[:, :, None],
                np.take(y, qry_idx, axis=1) * qry_mask[:, :, None], ctx_mask, qry_mask)
