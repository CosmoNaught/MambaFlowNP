"""Trains MambaFlowNP on a NOAA region."""

import argparse
import csv
import dataclasses
import json
import pathlib

import numpy as np

import mambaflownp as mf
from mambaflownp import noaa

VALIDATION_TASKS = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(noaa.TRAIN), required=True)
    parser.add_argument("--data", help="region file; data/noaa_<region>... by default")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    hp = noaa.TRAIN[args.region]
    cfg = noaa.MODELS[args.region]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    region = noaa.Region(args.data or noaa.data_path(args.region))
    sampler = region.sampler("train", seed=args.seed)

    def next_batch():
        return sampler.sample_padded(hp["batch_size"], cfg.max_points, hp["min_ctx"],
                                     hp["min_qry"])

    with open(out / "train_log.csv", "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["stage", "step", "loss", "seconds"])

        def log(stage, step, loss, seconds):
            writer.writerow([stage, step, f"{loss:.6f}", f"{seconds:.1f}"])
            log_file.flush()
            print(f"{stage:14s} {step:7d}  loss {loss:.4f}  {seconds:8.0f}s", flush=True)

        model = mf.train(cfg, next_batch, lr=hp["lr"], num_steps=hp["num_steps"],
                         distill_steps=hp["distill_steps"], n_couplings=hp["n_couplings"],
                         coupling_ode_steps=hp["coupling_ode_steps"], seed=args.seed, log=log)
        val = region.sampler("val", seed=args.seed)
        points = min(cfg.max_points, int(np.percentile(val.observed.sum(axis=1), 10)))
        validation = [val.sample_padded(hp["crps_batch_size"], points, hp["min_ctx"],
                                        hp["min_qry"])
                      for _ in range(VALIDATION_TASKS // hp["crps_batch_size"])]
        model = mf.crps_finetune(
            model, lambda: sampler.sample_padded(hp["crps_batch_size"], cfg.max_points,
                                                 hp["min_ctx"], hp["min_qry"]),
            validation, members=hp["crps_members"], seed=args.seed, log=log)

    mf.save_checkpoint(model, out / f"mambaflownp_{args.region}.npz")
    with open(out / "config.json", "w") as f:
        json.dump({"region": args.region, "seed": args.seed,
                   **dataclasses.asdict(cfg), **hp},
                  f, indent=2)


if __name__ == "__main__":
    main()
