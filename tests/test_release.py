"""Checks the released models against reference outputs, the consistency of samples across query
sets, and training end to end."""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import mambaflownp as mf
from mambaflownp import noaa

ROOT = pathlib.Path(__file__).resolve().parents[1]

REFERENCE = np.load(ROOT / "tests" / "reference_outputs.npz")
TOLERANCE = dict(rtol=1e-4, atol=1e-4)


def _reference(name):
    return {k.split("/", 1)[1]: v for k, v in REFERENCE.items() if k.startswith(name + "/")}


@pytest.mark.parametrize("region", ["california", "us"])
def test_matches_reference_outputs(region):
    ref = _reference(region)
    model = noaa.load_model(region)
    batch = ref["x_ctx"].shape[0]
    routing = model.precompute_routing(ref["x_ctx"], ref["x_qry"], mask=ref["mask"],
                                       offset=jnp.asarray(ref["offsets"][:batch]))
    velocity = model(ref["x_ctx"], ref["y_ctx"], ref["x_qry"], ref["y_t"], ref["t"], ref["c"],
                     routing_cache=routing, mask=ref["mask"])
    draws = mf.sample(model, ref["x_ctx"][:1], ref["y_ctx"][:1], ref["x_qry"][:1],
                      jnp.asarray(ref["keys"]), offsets=ref["offsets"])
    np.testing.assert_allclose(np.asarray(velocity), ref["velocity"], **TOLERANCE)
    np.testing.assert_allclose(np.asarray(draws), ref["draws"], **TOLERANCE)


def test_samples_do_not_depend_on_other_queries():
    ref = _reference("california")
    model = noaa.load_model("california")
    x_ctx, y_ctx, x_qry = ref["x_ctx"][:1], ref["y_ctx"][:1], ref["x_qry"][:1]
    keys = jnp.asarray(ref["keys"])
    together = mf.sample(model, x_ctx, y_ctx, x_qry, keys)
    apart = jnp.concatenate([mf.sample(model, x_ctx, y_ctx, x_qry[:, :30], keys),
                             mf.sample(model, x_ctx, y_ctx, x_qry[:, 30:], keys)], axis=1)
    np.testing.assert_allclose(np.asarray(apart), np.asarray(together), **TOLERANCE)


def test_training_round_trip(tmp_path):
    cfg = mf.Config(hidden_size=32, d_state=8, num_layers=2, rff_freqs=16, max_points=48)
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(60, 2)).astype(np.float32)
    values = np.sin(6 * coords[:, 0])[None] + 0.1 * rng.normal(size=(40, 60))
    values[rng.uniform(size=values.shape) < 0.1] = np.nan
    sampler = mf.FieldSampler(coords, values, seed=0)

    model = mf.train(cfg, lambda: sampler.sample_padded(4, 48, 8, 8), lr=1e-3, num_steps=3,
                     distill_steps=2, n_couplings=2, coupling_ode_steps=2, seed=0,
                     log=lambda *args: None)
    assert isinstance(model, mf.MambaFlowNP)
    model = mf.crps_finetune(model, lambda: sampler.sample_padded(2, 48, 8, 8),
                             [sampler.sample_padded(2, 48, 8, 8) for _ in range(2)], members=3,
                             max_steps=6, eval_every=2, patience=1, log=lambda *args: None)
    path = tmp_path / "model.npz"
    mf.save_checkpoint(model, path)
    restored = mf.load_checkpoint(mf.MambaFlowNP(cfg, rngs=nnx.Rngs(1)), path)

    x_ctx, y_ctx, x_qry, _ = sampler.sample(1, 20, 20)
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    a = mf.sample(model, jnp.asarray(x_ctx), jnp.asarray(y_ctx), jnp.asarray(x_qry), keys)
    b = mf.sample(restored, jnp.asarray(x_ctx), jnp.asarray(y_ctx), jnp.asarray(x_qry), keys)
    assert np.all(np.isfinite(a))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
