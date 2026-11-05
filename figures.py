"""Renders the figures in results/ from the released weights and the evaluation results."""

import csv
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import mambaflownp as mf
from mambaflownp import noaa

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = RESULTS / "qualitative"
sys.path.insert(0, str(ROOT / "plotting"))

import maps  # noqa: E402
import pareto  # noqa: E402

# Map selections: (eval seed, pinned date, context fraction, draws per call).
MAPS = {"utah": (7, None, 0.25, None),
        "california": (7, None, 0.25, None),
        "us": (11, 20200719, 0.10, 16)}
# Milliseconds per task (128 samples) on one NVIDIA L40S.
LATENCY_MS = {"utah": (43.3, 0.4), "california": (67.9, 0.5), "us": (1125.9, 0.1)}
NAMES = {"FlowNP": "FNP"}


def dump_path(name):
    return DATA / f"{name}_f{round(MAPS[name][2] * 100):02d}.npz"


def dump_map(name):
    """Samples the released model on one test day and saves the stations and predictions."""
    eval_seed, date_int, frac, chunk = MAPS[name]
    region = noaa.Region(noaa.data_path(name))
    model = noaa.load_model(name)
    total = noaa.MODELS[name].max_points
    observed = np.isfinite(region.values)
    test = region.split["test"]
    eligible = test[observed[test].sum(axis=1) >= total]
    if date_int is None:
        rng = np.random.default_rng(eval_seed)
        day = int(rng.choice(eligible))
    else:
        day = int(eligible[np.flatnonzero(region.dates[eligible] == date_int)[0]])
        rng = np.random.default_rng(eval_seed * 9001 + day)
    chosen = rng.choice(np.flatnonzero(observed[day]), size=total, replace=False)
    chosen = chosen[rng.permutation(total)]
    values = region.values[day, chosen]

    n_ctx = max(1, min(total - 1, round(total * frac)))
    keys = jax.random.split(jax.random.PRNGKey(eval_seed * 100000 + round(frac * 1000)), 128)
    step = chunk or 128
    x_ctx = jnp.asarray(region.coords[chosen[:n_ctx]][None])
    y_ctx = jnp.asarray(region.normalizer.transform(values[:n_ctx])[None, :, None])
    x_qry = jnp.asarray(region.coords[chosen[n_ctx:]][None])
    z = np.concatenate([np.asarray(mf.sample(model, x_ctx, y_ctx, x_qry, keys[i:i + step]))
                        for i in range(0, 128, step)])[:, :, 0]
    samples = region.normalizer.inverse(z)
    mean = samples.mean(axis=0)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(dump_path(name),
             samples=samples, pred_mean=mean, sq_err=(mean - values[n_ctx:]) ** 2,
             xc_lonlat=region.lonlat[chosen[:n_ctx]], yc_raw=values[:n_ctx],
             xt_lonlat=region.lonlat[chosen[n_ctx:]], yt_raw=values[n_ctx:],
             date_int=np.int32(region.dates[day]), region=np.str_(name),
             ctx_fraction=np.float32(frac))


def mean_se(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def pareto_results(name):
    """Mean and standard error of RMSE, CRPS and time per task of each model on one region."""
    with open(RESULTS / "baselines.csv") as f:
        results = {NAMES.get(b["model"], b["model"]): {
            "rmse": (float(b["rmse_mean"]), float(b["rmse_se"])),
            "crps": (float(b["crps_mean"]), float(b["crps_se"])),
            "time_ms": (float(b["time_ms_mean"]), float(b["time_ms_se"]))}
            for b in csv.DictReader(f) if b["region"] == name}
    rows = []
    for path in sorted((RESULTS / name).glob("seed_*.csv")):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    results["MFNP"] = {"rmse": mean_se(np.sqrt([float(r["mse"]) for r in rows])),
                       "crps": mean_se([float(r["crps"]) for r in rows]),
                       "time_ms": LATENCY_MS[name]}
    return results


def main():
    for name in MAPS:
        if not dump_path(name).exists():
            dump_map(name)
        maps.render_map(name, dump_path(name), RESULTS / f"noaa_qualitative_{name}")
        pareto.plot(pareto_results(name), name, RESULTS / f"pareto_noaa_{name}")


if __name__ == "__main__":
    main()
