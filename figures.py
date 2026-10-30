"""Renders the figures in results/ from the released weights."""

import csv
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import mambaflownp as mf
from mambaflownp import noaa

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = RESULTS / "qualitative"
sys.path.insert(0, str(ROOT / "plotting"))

# Figure selections: (eval seed, pinned date, context fractions, draws per call).
QUALITATIVE = {"utah": (7, None, (0.25, 0.50, 0.75), None),
               "california": (7, None, (0.25, 0.50, 0.75), None),
               "us": (11, 20200719, (0.10,), 16)}
LABELS = {"utah": r"NOAA Utah ($K{=}222$)", "california": r"NOAA California ($K{=}486$)",
          "us": r"NOAA US ($K{=}5{,}580$)"}
# Milliseconds per task (128 samples) on one NVIDIA L40S.
LATENCY_MS = {"utah": (28.8, 0.2), "california": (62.3, 0.1), "us": (1161.0, 2.0)}
MODEL_KEYS = {"CNP": "cnp", "NP": "np", "CANP": "canp", "ANP": "anp", "TNPA": "tnpa",
              "NDP": "ndp", "FlowNP": "fnp"}


def dump_qualitative(name):
    eval_seed, date_int, fractions, chunk = QUALITATIVE[name]
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

    DATA.mkdir(parents=True, exist_ok=True)
    for frac in fractions:
        n_ctx = max(1, min(total - 1, int(round(total * frac))))
        seed = eval_seed * 100000 + int(round(frac * 1000))
        keys = jax.random.split(jax.random.PRNGKey(seed), 128)
        step = chunk or 128
        x_ctx = jnp.asarray(region.coords[chosen[:n_ctx]][None])
        y_ctx = jnp.asarray(region.normalizer.transform(values[:n_ctx])[None, :, None])
        x_qry = jnp.asarray(region.coords[chosen[n_ctx:]][None])
        z = np.concatenate([np.asarray(mf.sample(model, x_ctx, y_ctx, x_qry, keys[i:i + step]))
                            for i in range(0, 128, step)])[:, :, 0]
        samples = region.normalizer.inverse(z)
        mean = samples.mean(axis=0)
        np.savez(DATA / f"{name}_f{int(round(frac * 100)):02d}.npz",
                 samples=samples, pred_mean=mean, sq_err=(mean - values[n_ctx:]) ** 2,
                 xc_lonlat=region.lonlat[chosen[:n_ctx]], yc_raw=values[:n_ctx],
                 xt_lonlat=region.lonlat[chosen[n_ctx:]], yt_raw=values[n_ctx:],
                 date_int=np.int32(region.dates[day]), region=np.str_(name),
                 ctx_fraction=np.float32(frac))


def pareto_tiers():
    with open(RESULTS / "baselines.csv") as f:
        baselines = list(csv.DictReader(f))
    tiers = {}
    for name in ("utah", "california", "us"):
        rows = []
        for path in sorted((RESULTS / name).glob("seed_*.csv")):
            with open(path) as f:
                rows.extend(csv.DictReader(f))
        rmse = np.sqrt([float(r["mse"]) for r in rows])
        crps = np.array([float(r["crps"]) for r in rows])
        models = {MODEL_KEYS[b["model"]]: {
            "rmse": (float(b["rmse_mean"]), float(b["rmse_se"])),
            "crps": (float(b["crps_mean"]), float(b["crps_se"])),
            "time_ms": (float(b["time_ms_mean"]), float(b["time_ms_se"]))}
            for b in baselines if b["region"] == name}
        models["mfnp"] = {
            "rmse": (float(rmse.mean()), float(rmse.std(ddof=1) / np.sqrt(len(rmse)))),
            "crps": (float(crps.mean()), float(crps.std(ddof=1) / np.sqrt(len(crps)))),
            "time_ms": LATENCY_MS[name]}
        tiers[name] = {"label": LABELS[name], "models": models}
    return tiers


def main():
    for name in QUALITATIVE:
        dump_qualitative(name)

    import plot_noaa_qualitative as qualitative
    import plot_pareto_noaa_per_tier as pareto

    qualitative.DATA_DIR = DATA
    qualitative.OUT_DIR = RESULTS
    qualitative.render("utah", fractions=(0.25, 0.50, 0.75))
    qualitative.render("california", fractions=(0.25, 0.50, 0.75))
    qualitative.render("us", fractions=(0.10,), panels=("context", "error"))
    for name, tier in pareto_tiers().items():
        pareto.plot_one_tier(name, tier, RESULTS / f"pareto_noaa_{name}")


if __name__ == "__main__":
    main()
