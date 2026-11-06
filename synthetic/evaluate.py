"""Evaluates a MambaFlowNP checkpoint on the cached test episodes of a synthetic task.

MSE, CRPS and coverage are averaged over the 256 episodes of a seed. C2ST compares the first
episode's samples with exact GP posterior draws. Latency is the median of 10 timed 128-draw calls
after 5 warm-up calls.
"""

import argparse
import csv
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import mambaflownp as mf
from synthetic import gp


def make_draw(model, name, ode_steps=1, heun=False):
    """Returns draw(x_ctx, y_ctx, x_qry, seed, num_draws) -> samples (num_draws, n_qry)."""

    def draw(x_ctx, y_ctx, x_qry, seed, num_draws):
        keys = jax.random.split(jax.random.PRNGKey(seed), num_draws)
        return mf.sample_views(model, gp.to_unit(name, x_ctx)[None], jnp.asarray(y_ctx)[None],
                               gp.to_unit(name, x_qry)[None], keys,
                               ode_steps=ode_steps, heun=heun)[:, :, 0]

    return draw


def time_ms(fn, warmup=5, reps=10):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def evaluate_seed(model, name, eval_seed, ode_steps=1, heun=False, time_only=False):
    """Returns the metrics of one evaluation seed, or only its latency if ``time_only``."""
    ep = gp.load_episodes(name, eval_seed)
    num_draws = gp.EVAL[name]["num_draws"]
    draw = make_draw(model, name, ode_steps, heun)
    if time_only:
        return dict(time_ms=time_ms(lambda: draw(ep["x_ctx"][0], ep["y_ctx"][0], ep["x_qry"][0],
                                                 eval_seed * 100_000 + 17, num_draws))), 1
    rows = []
    for i in range(len(ep["x_ctx"])):
        truth = ep["y_qry"][i, :, 0]
        samples = np.asarray(draw(ep["x_ctx"][i], ep["y_ctx"][i], ep["x_qry"][i],
                                  eval_seed * 100_000 + i, num_draws))
        rows.append((float(np.mean((samples.mean(axis=0) - truth) ** 2)),
                     mf.crps(samples, truth),
                     mf.interval_coverage(samples, truth, 0.90),
                     mf.interval_coverage(samples, truth, 0.95)))
        if i == 0:
            reference = gp.gp_posterior_samples(
                ep["x_ctx"][0], ep["y_ctx"][0], ep["x_qry"][0], gp.TASKS[name]["kernel"],
                ep["length"][0], ep["scale"][0], num_draws, seed=eval_seed)
            c2st = mf.c2st(samples, reference, seed=eval_seed)
            latency = time_ms(lambda: draw(ep["x_ctx"][0], ep["y_ctx"][0], ep["x_qry"][0],
                                           eval_seed * 100_000 + 17, num_draws))
    mse, crps, cov90, cov95 = np.mean(rows, axis=0).tolist()
    return dict(mse=mse, crps=crps, coverage_90=cov90, coverage_95=cov95, c2st=c2st,
                time_ms=latency), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(gp.TASKS), required=True)
    parser.add_argument("--weights", required=True, help="checkpoint from synthetic/train.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--variant", choices=sorted(gp.VARIANTS), default="full")
    parser.add_argument("--ode-steps", type=int, default=1)
    parser.add_argument("--heun", action="store_true", help="Heun instead of Euler steps")
    parser.add_argument("--time-only", action="store_true",
                        help="measure only the per-episode latency")
    args = parser.parse_args()

    model = gp.load_model(args.task, weights=args.weights, variant=args.variant)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        metrics, num_tasks = evaluate_seed(model, args.task, seed, args.ode_steps,
                                           args.heun, args.time_only)
        with open(out / f"seed_{seed}.csv", "w", newline="") as f:
            fields = dict(seed=seed, tasks=num_tasks, variant=args.variant,
                          solver="heun" if args.heun else "euler", ode_steps=args.ode_steps)
            writer = csv.DictWriter(f, fieldnames=[*fields, *metrics])
            writer.writeheader()
            writer.writerow({**fields, **metrics})
        print(f"seed {seed}: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()),
              flush=True)


if __name__ == "__main__":
    main()
