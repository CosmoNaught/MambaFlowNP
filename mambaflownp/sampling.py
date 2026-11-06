"""Posterior sampling and teacher couplings."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .model import LATENT


def random_offsets(rng, num, dim: int = 2):
    """Scan-grid offsets, shape (num, dim), uniform in [0, 0.5)."""
    return rng.uniform(0.0, 0.5, size=(num, dim)).astype(np.float32)


def base(model, keys, x_qry, qry_mask):
    """Base noise ``model.base_noise(c, x)`` with one latent c per key, shape
    (len(keys), n_qry, 1), and the latents."""
    c = jax.vmap(lambda k: jax.random.normal(k, (LATENT,)))(keys)
    return model.base_noise(c, x_qry) * qry_mask[:, :, None], c


def velocity(model, x_ctx, y_ctx, x_qry, f, t, c, routing, mask):
    """Velocity at the queries."""
    return model(x_ctx, y_ctx, x_qry, f, t, c, routing_cache=routing, mask=mask)


@partial(nnx.jit, static_argnums=(8,))
def _euler(model, x_ctx, y_ctx, x_qry, keys, ctx_mask, qry_mask, offset, ode_steps):
    num_draws = keys.shape[0]
    x_ctx, y_ctx, x_qry = (jnp.tile(a, (num_draws, 1, 1)) for a in (x_ctx, y_ctx, x_qry))
    ctx_mask, qry_mask = jnp.tile(ctx_mask, (num_draws, 1)), jnp.tile(qry_mask, (num_draws, 1))
    mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)
    if offset is not None:
        offset = jnp.tile(offset, (num_draws, 1))
    routing = model.precompute_routing(x_ctx, x_qry, mask=mask, offset=offset)

    f, c = base(model, keys, x_qry, qry_mask)
    dt = 1.0 / ode_steps

    def step(f, i):
        t = jnp.full((num_draws,), jnp.maximum(i * dt, 1e-4))
        return f + dt * velocity(model, x_ctx, y_ctx, x_qry, f, t, c, routing, mask), None

    f, _ = jax.lax.scan(step, f, jnp.arange(ode_steps, dtype=jnp.float32))
    return f


def sample(model, x_ctx, y_ctx, x_qry, keys, ode_steps: int = 1, ctx_mask=None, qry_mask=None,
           offsets=None, draws_per_call=None):
    """Draws one posterior sample per key for a single task.

    Inputs have a leading batch dimension of 1. Returns shape (num_keys, n_qry, 1).

    The same keys give the same samples for any set of queries, whole or in chunks.

    A model with ``scan_offsets`` > 0 splits the keys as evenly as possible over that many
    scan-grid offsets, drawn from the first key unless ``offsets`` (num_offsets, 2) is given.
    ``draws_per_call`` bounds the draws evaluated at once.
    """
    if ctx_mask is None:
        ctx_mask = jnp.ones(x_ctx.shape[:2], dtype=bool)
    if qry_mask is None:
        qry_mask = jnp.ones(x_qry.shape[:2], dtype=bool)
    if offsets is None and model.scan_offsets:
        seed = int(jax.random.randint(keys[0], (), 0, 2**31 - 1))
        offsets = random_offsets(np.random.default_rng(seed), model.scan_offsets,
                                 model.coord_dim)
    groups = [None] if offsets is None else [jnp.asarray(o)[None] for o in offsets]
    bounds = np.linspace(0, len(keys), len(groups) + 1).round().astype(int)
    parts = []
    for offset, start, stop in zip(groups, bounds[:-1], bounds[1:]):
        step = draws_per_call or max(stop - start, 1)
        parts += [_euler(model, x_ctx, y_ctx, x_qry, keys[i:min(i + step, stop)], ctx_mask,
                         qry_mask, offset, ode_steps) for i in range(start, stop, step)]
    return jnp.concatenate(parts)


@partial(nnx.jit, static_argnums=(8, 9))
def _integrate_views(model, x_ctx, y_ctx, x_qry, keys, ctx_mask, qry_mask, offset, ode_steps,
                     heun):
    """Euler or Heun sampling where draw i sees its own coordinates x_ctx[i], x_qry[i] and
    offset[i]."""
    num_draws = keys.shape[0]
    y_ctx = jnp.tile(y_ctx, (num_draws, 1, 1))
    ctx_mask, qry_mask = jnp.tile(ctx_mask, (num_draws, 1)), jnp.tile(qry_mask, (num_draws, 1))
    mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)
    routing = model.precompute_routing(x_ctx, x_qry, mask=mask, offset=offset)
    f, c = base(model, keys, x_qry, qry_mask)
    dt = 1.0 / ode_steps

    def v(f, t):
        return velocity(model, x_ctx, y_ctx, x_qry, f, jnp.full((num_draws,), t), c, routing,
                        mask)

    def step(f, i):
        t1 = jnp.maximum(i * dt, 1e-4)
        v1 = v(f, t1)
        if not heun:
            return f + dt * v1, None
        t2 = jnp.minimum((i + 1.0) * dt, 1.0 - 1e-4)
        return f + (dt / 2.0) * (v1 + v(f + dt * v1, t2)), None

    f, _ = jax.lax.scan(step, f, jnp.arange(ode_steps, dtype=jnp.float32))
    return f


def sample_views(model, x_ctx, y_ctx, x_qry, keys, ode_steps: int = 1, heun: bool = False,
                 draws_per_call=None):
    """Draws one posterior sample per key.

    The keys cycle over the model's scan-grid offsets. Inputs have a leading batch dimension of
    1. Returns shape (num_keys, n_qry, 1).
    """
    num = len(keys)
    seed = int(jax.random.randint(keys[0], (), 0, 2**31 - 1))
    x_ctx, x_qry = np.asarray(x_ctx)[0], np.asarray(x_qry)[0]
    x_ctx_v = np.broadcast_to(x_ctx, (num, *x_ctx.shape))
    x_qry_v = np.broadcast_to(x_qry, (num, *x_qry.shape))
    offset_v = None
    if model.scan_offsets:
        offsets = random_offsets(np.random.default_rng(seed), model.scan_offsets, model.coord_dim)
        offset_v = jnp.asarray(offsets[np.arange(num) % model.scan_offsets])
    ctx_mask = jnp.ones((1, x_ctx.shape[0]), dtype=bool)
    qry_mask = jnp.ones((1, x_qry.shape[0]), dtype=bool)
    step = draws_per_call or num
    parts = [_integrate_views(model, jnp.asarray(x_ctx_v[i:i + step]), y_ctx,
                              jnp.asarray(x_qry_v[i:i + step]), keys[i:i + step], ctx_mask,
                              qry_mask, None if offset_v is None else offset_v[i:i + step],
                              ode_steps, heun)
             for i in range(0, num, step)]
    return jnp.concatenate(parts)


@partial(nnx.jit, static_argnums=(8,))
def teacher_couplings(model, x_ctx, y_ctx, x_qry, keys, ctx_mask, qry_mask, offset, ode_steps):
    """Integrates the teacher with Heun's method and returns (noise, endpoint, latent)."""
    batch = x_qry.shape[0]
    mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)
    routing = model.precompute_routing(x_ctx, x_qry, mask=mask, offset=offset)
    f0, c = base(model, keys, x_qry, qry_mask)
    dt = 1.0 / ode_steps

    def v(f, t):
        return velocity(model, x_ctx, y_ctx, x_qry, f, t, c, routing, mask)

    def step(f, i):
        t1 = jnp.full((batch,), jnp.maximum(i * dt, 1e-4))
        t2 = jnp.full((batch,), jnp.minimum((i + 1.0) * dt, 1.0 - 1e-4))
        v1 = v(f, t1)
        v2 = v(f + dt * v1, t2)
        return f + (dt / 2.0) * (v1 + v2), None

    f1, _ = jax.lax.scan(step, f0, jnp.arange(ode_steps, dtype=jnp.float32))
    return f0, f1, c
