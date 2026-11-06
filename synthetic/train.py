"""Trains MambaFlowNP on a synthetic Gaussian-process task."""

import argparse
import csv
import dataclasses
import json
import pathlib

import mambaflownp as mf
from synthetic import gp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(gp.TASKS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variant", choices=sorted(gp.VARIANTS), default="full")
    args = parser.parse_args()

    hp = gp.TRAIN[args.task]
    cfg = gp.config(args.task, args.variant)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sampler = gp.TaskSampler(args.task, seed=args.seed)

    def next_batch():
        return sampler.sample_padded(hp["batch_size"], hp["min_ctx"], hp["min_qry"])

    with open(out / "train_log.csv", "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["stage", "step", "loss", "seconds"])

        def log(stage, step, loss, seconds):
            writer.writerow([stage, step, f"{loss:.6f}", f"{seconds:.1f}"])
            log_file.flush()
            print(f"{stage:14s} {step:7d}  loss {loss:.4f}  {seconds:8.0f}s", flush=True)

        model = mf.train(cfg, next_batch, lr=hp["lr"], num_steps=hp["num_steps"],
                         distill_steps=hp["distill_steps"], n_couplings=hp["n_couplings"],
                         coupling_ode_steps=hp["coupling_ode_steps"], seed=args.seed, log=log,
                         on_teacher=lambda teacher: mf.save_checkpoint(
                             teacher, out / f"teacher_{args.task}.npz"))

    mf.save_checkpoint(model, out / f"mambaflownp_{args.task}.npz")
    with open(out / "config.json", "w") as f:
        json.dump({"task": args.task, "variant": args.variant, "seed": args.seed,
                   **dataclasses.asdict(cfg), **hp},
                  f, indent=2)


if __name__ == "__main__":
    main()
