"""Evaluates a MambaFlowNP checkpoint on the test split of a NOAA region."""

import argparse
import csv
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

import mambaflownp as mf
from mambaflownp import noaa

METRICS = ("mse", "crps", "coverage_90", "coverage_95")


def draw(model, x_ctx, y_ctx, x_qry, seed, num_draws, draws_per_call=None):
    """Returns posterior samples for one task, shape (num_draws, n_qry)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), num_draws)
    draws = mf.sample(model, jnp.asarray(x_ctx), jnp.asarray(y_ctx), jnp.asarray(x_qry), keys,
                      draws_per_call=draws_per_call)
    return np.asarray(draws)[:, :, 0]


def test_tasks(region, name, eval_seed):
    """Yields (x_ctx, t_ctx, x_qry, t_qry, draw_seed) for each test task of one seed."""
    protocol, n_points = noaa.EVAL[name], noaa.MODELS[name].max_points
    sampler = mf.FieldSampler(region.coords, region.values[region.split["test"]], seed=eval_seed)
    for _ in range(protocol["num_batches"]):
        x_ctx, t_ctx, x_qry, t_qry = sampler.sample(protocol["batch_size"], n_points // 2,
                                                    n_points - n_points // 2)
        for i in range(protocol["batch_size"]):
            yield (x_ctx[i:i + 1], t_ctx[i:i + 1], x_qry[i:i + 1], t_qry[i, :, 0],
                   eval_seed * 100_000 + i)


def evaluate_seed(model, region, name, eval_seed, draws_per_call=None):
    """Returns the mean of each metric over the test tasks of one seed."""
    num_draws = noaa.EVAL[name]["num_draws"]
    rows = []
    for x_ctx, t_ctx, x_qry, truth, seed in test_tasks(region, name, eval_seed):
        z = draw(model, x_ctx, region.normalizer.transform(t_ctx), x_qry, seed, num_draws,
                 draws_per_call)
        samples = region.normalizer.inverse(z)
        rows.append((float(np.mean((samples.mean(axis=0) - truth) ** 2)),
                     mf.crps(samples, truth),
                     mf.interval_coverage(samples, truth, 0.90),
                     mf.interval_coverage(samples, truth, 0.95)))
    return dict(zip(METRICS, np.mean(rows, axis=0).tolist())), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(noaa.MODELS), required=True)
    parser.add_argument("--data", help="region file; data/noaa_<region>... by default")
    parser.add_argument("--weights", help="checkpoint; the released weights by default")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--draws-per-call", type=int, default=None)
    args = parser.parse_args()

    region = noaa.Region(args.data or noaa.data_path(args.region))
    model = noaa.load_model(args.region, weights=args.weights)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        metrics, num_tasks = evaluate_seed(model, region, args.region, seed, args.draws_per_call)
        with open(out / f"seed_{seed}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "tasks", *metrics])
            writer.writeheader()
            writer.writerow({"seed": seed, "tasks": num_tasks, **metrics})
        print(f"seed {seed}: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()),
              flush=True)


if __name__ == "__main__":
    main()
