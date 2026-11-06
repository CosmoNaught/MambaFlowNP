"""Flow-matching training, reflow distillation and CRPS fine-tuning."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .model import Config, MambaFlowNP
from .sampling import base, random_offsets, teacher_couplings, velocity


EMA_DECAY = 0.9995


def make_optimizer(model, lr: float, num_steps: int):
    """AdamW with warmup-cosine schedule and gradient clipping. Steps with non-finite gradients
    are skipped; after 100 in a row the update is applied so that divergence is not hidden."""
    if num_steps <= 1:
        schedule = optax.constant_schedule(lr)
    else:
        warmup = max(0, min(100, num_steps - 1))
        schedule = optax.warmup_cosine_decay_schedule(1e-5, lr, warmup, num_steps, 1e-5)
    return nnx.Optimizer(
        model, optax.apply_if_finite(
            optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule)), 100),
        wrt=nnx.Param)


def skipped_steps(optimizer) -> int:
    """Number of updates skipped because their gradients were not finite."""
    leaves = jax.tree_util.tree_leaves_with_path(nnx.state(optimizer))
    return sum(int(v) for path, v in leaves if "total_notfinite" in jax.tree_util.keystr(path))


def _masked_mse(model, x_ctx, y_ctx, x_qry, f_t, t, c, target, ctx_mask, qry_mask, offset):
    mask = jnp.concatenate([ctx_mask, qry_mask], axis=1)
    routing = model.precompute_routing(x_ctx, x_qry, mask=mask, offset=offset)
    err = (velocity(model, x_ctx, y_ctx, x_qry, f_t, t, c, routing, mask) - target) ** 2
    return jnp.sum(err * qry_mask[:, :, None]) / jnp.maximum(jnp.sum(qry_mask), 1.0)


def _update(model, optimizer, ema, loss_fn):
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    ema = optax.incremental_update(nnx.state(model, nnx.Param), ema, 1.0 - EMA_DECAY)
    return loss, ema


@nnx.jit
def cfm_step(model, optimizer, ema, key, x_ctx, y_ctx, x_qry, y_qry, ctx_mask, qry_mask,
             offset=None):
    """Performs one flow-matching step. Returns the loss and updated EMA parameters."""
    key_t, key_n = jax.random.split(key)
    t = jax.random.uniform(key_t, (x_ctx.shape[0],), dtype=jnp.float32)
    f0, c = base(model, jax.random.split(key_n, x_ctx.shape[0]), x_qry, qry_mask)
    f0 = f0 * qry_mask[:, :, None]
    f_t = (1.0 - t[:, None, None]) * f0 + t[:, None, None] * y_qry
    return _update(model, optimizer, ema, lambda m: _masked_mse(
        m, x_ctx, y_ctx, x_qry, f_t, t, c, y_qry - f0, ctx_mask, qry_mask, offset))


@nnx.jit
def reflow_step(model, optimizer, ema, key, x_ctx, y_ctx, x_qry, f0, f1, c, ctx_mask, qry_mask,
                offset=None):
    """Performs one reflow step on a teacher coupling (f0, f1, c). Returns the loss and updated
    EMA parameters."""
    t = jax.random.uniform(key, (x_ctx.shape[0],), dtype=jnp.float32)
    f_t = (1.0 - t[:, None, None]) * f0 + t[:, None, None] * f1
    return _update(model, optimizer, ema, lambda m: _masked_mse(
        m, x_ctx, y_ctx, x_qry, f_t, t, c, f1 - f0, ctx_mask, qry_mask, offset))


def _as_jnp(batch):
    return tuple(jnp.asarray(a, dtype=bool if a.dtype == bool else jnp.float32) for a in batch)


def train(cfg: Config, next_batch, *, lr: float, num_steps: int, distill_steps: int,
          n_couplings: int, coupling_ode_steps: int = 10, seed: int = 1, log=None,
          log_every: int = 200, on_teacher=None):
    """Trains a teacher with flow matching and distils it with reflow.

    Args:
      cfg: Model configuration.
      next_batch: Callable returning (x_ctx, y_ctx, x_qry, y_qry, ctx_mask, qry_mask).
      lr: Peak learning rate.
      num_steps: Flow-matching steps.
      distill_steps: Reflow steps. If 0, the teacher is returned.
      n_couplings: Number of teacher coupling batches.
      coupling_ode_steps: Heun steps for the teacher couplings.
      seed: Random seed.
      log: Callable taking (stage, step, loss, seconds). Prints by default.
      on_teacher: Optional callable given the trained teacher before distillation.

    Returns:
      The trained model with EMA weights.
    """
    emit = log or (lambda stage, step, loss, secs: print(stage, step, f"{loss:.4f}", f"{secs:.0f}"))
    rng = np.random.default_rng(seed)

    def batch():
        b = _as_jnp(next_batch())
        offset = (jnp.asarray(random_offsets(rng, b[0].shape[0], cfg.coord_dim)) if cfg.scan_offsets
                  else None)
        return (*b, offset)

    model = MambaFlowNP(cfg, rngs=nnx.Rngs(seed))
    optimizer = make_optimizer(model, lr, num_steps)
    ema = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
    key = jax.random.PRNGKey(seed)
    start = time.time()

    for step in range(1, num_steps + 1):
        key, step_key = jax.random.split(key)
        loss, ema = cfm_step(model, optimizer, ema, step_key, *batch())
        if step % log_every == 0 or step == num_steps:
            emit("flow_matching", step, float(loss), time.time() - start)
    emit("skipped", skipped_steps(optimizer), float("nan"), time.time() - start)
    nnx.update(model, ema)
    if on_teacher is not None:
        on_teacher(model)
    if distill_steps <= 0:
        return model

    couplings = []
    coupling_key = jax.random.PRNGKey(seed + 100)
    for i in range(n_couplings):
        x_ctx, y_ctx, x_qry, _, ctx_mask, qry_mask, offset = batch()
        coupling_key, traj_key = jax.random.split(coupling_key)
        keys = jax.random.split(traj_key, x_ctx.shape[0])
        f0, f1, c = teacher_couplings(model, x_ctx, y_ctx, x_qry, keys, ctx_mask, qry_mask,
                                      offset, coupling_ode_steps)
        couplings.append((x_ctx, y_ctx, x_qry, f0, f1, c, ctx_mask, qry_mask, offset))
        if (i + 1) % log_every == 0 or i + 1 == n_couplings:
            emit("couplings", i + 1, float("nan"), time.time() - start)

    student = MambaFlowNP(cfg, rngs=nnx.Rngs(seed + 1))
    nnx.update(student, nnx.state(model, nnx.Param))
    optimizer = make_optimizer(student, lr, distill_steps)
    ema = jax.tree.map(lambda x: x.copy(), nnx.state(student, nnx.Param))
    for step in range(1, distill_steps + 1):
        key, step_key = jax.random.split(key)
        loss, ema = reflow_step(student, optimizer, ema, step_key,
                                *couplings[(step - 1) % n_couplings])
        if step % log_every == 0 or step == distill_steps:
            emit("reflow", step, float(loss), time.time() - start)
    emit("skipped", skipped_steps(optimizer), float("nan"), time.time() - start)
    nnx.update(student, ema)
    return student


def _fair_crps(model, key, x_ctx, y_ctx, x_qry, y_qry, ctx_mask, qry_mask, offset, members):
    """Mean over query points of E|X - y| - E|X - X'| / 2 from ``members`` one-step samples."""
    m = members
    tile = lambda a: jnp.tile(a, (m,) + (1,) * (a.ndim - 1))  # noqa: E731
    xc, yc, xq, cm, qm = map(tile, (x_ctx, y_ctx, x_qry, ctx_mask, qry_mask))
    off = None if offset is None else tile(offset)
    mask = jnp.concatenate([cm, qm], axis=1)
    f0, c = base(model, jax.random.split(key, xq.shape[0]), xq, qm)
    f0 = f0 * qm[:, :, None]
    routing = model.precompute_routing(xc, xq, mask=mask, offset=off)
    t = jnp.full((xq.shape[0],), 1e-4)

    def loss_fn(mdl):
        draws = f0 + velocity(mdl, xc, yc, xq, f0, t, c, routing, mask)
        draws = draws.reshape(m, -1, *draws.shape[1:])
        skill = jnp.mean(jnp.abs(draws - y_qry[None]), axis=0)
        spread = jnp.sum(jnp.abs(draws[:, None] - draws[None]), axis=(0, 1)) / (2 * m * (m - 1))
        per_point = (skill - spread)[..., 0] * qry_mask
        return jnp.sum(per_point) / jnp.maximum(jnp.sum(qry_mask), 1.0)

    return loss_fn


def crps_finetune(model, next_batch, validation, *, members: int, lr: float = 1e-4,
                  max_steps: int = 40_000, eval_every: int = 1_000, patience: int = 5,
                  seed: int = 1, log=None, log_every: int = 200):
    """Fine-tunes a one-step sampler on the fair CRPS of its own samples, with early stopping.

    The CRPS on ``validation`` (a list of batches) is measured every ``eval_every`` steps with
    fixed noise; training stops after ``patience`` checks without improvement and the best
    weights, including the starting ones, are returned. ``lr`` applies to batches of 8 tasks and
    scales with the square root of the batch size.

    Returns:
      The fine-tuned model with the EMA weights that scored best on ``validation``.
    """
    emit = log or (lambda stage, step, loss, secs: print(stage, step, f"{loss:.4f}", f"{secs:.0f}"))

    @nnx.jit
    def step(model, optimizer, ema, key, *batch):
        return _update(model, optimizer, ema, _fair_crps(model, key, *batch, members))

    @nnx.jit
    def score(model, key, *batch):
        return _fair_crps(model, key, *batch, members)(model)

    def offsets_for(rng, batch):
        return (jnp.asarray(random_offsets(rng, batch[0].shape[0], model.coord_dim)) if model.scan_offsets
                else None)

    val_rng = np.random.default_rng(seed + 900)
    validation = [(*_as_jnp(b), offsets_for(val_rng, b)) for b in validation]
    val_keys = jax.random.split(jax.random.PRNGKey(seed + 900), len(validation))
    evaluator = nnx.clone(model)

    def validate(params):
        nnx.update(evaluator, params)
        return float(np.mean([score(evaluator, k, *b) for k, b in zip(val_keys, validation)]))

    rng = np.random.default_rng(seed)
    batch_size = validation[0][0].shape[0]
    optimizer = make_optimizer(model, lr * (batch_size / 8) ** 0.5, max_steps)
    ema = jax.tree.map(lambda x: x.copy(), nnx.state(model, nnx.Param))
    best, best_params, stale = validate(ema), jax.tree.map(lambda x: x.copy(), ema), 0
    key = jax.random.PRNGKey(seed + 500)
    start = time.time()
    emit("crps_val", 0, best, 0.0)
    for i in range(1, max_steps + 1):
        b = _as_jnp(next_batch())
        key, k = jax.random.split(key)
        loss, ema = step(model, optimizer, ema, k, *b, offsets_for(rng, b))
        if i % log_every == 0:
            emit("crps", i, float(loss), time.time() - start)
        if i % eval_every == 0:
            current = validate(ema)
            emit("crps_val", i, current, time.time() - start)
            if current < best:
                best, best_params, stale = current, jax.tree.map(lambda x: x.copy(), ema), 0
            else:
                stale += 1
                if stale >= patience:
                    break
    emit("skipped", skipped_steps(optimizer), float("nan"), time.time() - start)
    if not np.isfinite(best):
        raise FloatingPointError("validation CRPS was never finite")
    nnx.update(model, best_params)
    return model
