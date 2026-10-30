"""NOAA GHCN-Daily daily maximum temperature regions and their model, training and evaluation
settings."""

import pathlib

import numpy as np
from flax import nnx

from .data import FieldSampler, QuantileNormalizer
from .model import Config, MambaFlowNP, load_checkpoint

MODELS = {
    "california": Config(hidden_size=256, d_state=16, num_layers=6, rff_freqs=192,
                         max_points=486, c_scale=0.25),
    "us": Config(hidden_size=256, d_state=16, num_layers=6, rff_freqs=384, max_points=5580,
                 c_scale=0.25),
    "utah": Config(hidden_size=128, d_state=16, num_layers=6, rff_freqs=256, max_points=222,
                   c_scale=0.25),
}

TRAIN = {
    "california": dict(lr=0.0015174725019113822, batch_size=16, num_steps=80_000,
                       distill_steps=20_000, n_couplings=5_000, coupling_ode_steps=10,
                       min_ctx=16, min_qry=16, crps_batch_size=8, crps_members=8),
    "us": dict(lr=1e-3, batch_size=8, num_steps=80_000, distill_steps=20_000,
               n_couplings=1_000, coupling_ode_steps=10, min_ctx=16, min_qry=16,
               crps_batch_size=2, crps_members=4),
    "utah": dict(lr=1e-3, batch_size=16, num_steps=80_000, distill_steps=20_000, n_couplings=5_000,
                 coupling_ode_steps=10, min_ctx=16, min_qry=16, crps_batch_size=8, crps_members=8),
}

EVAL = {
    "california": dict(num_batches=64, batch_size=16, num_draws=128),
    "us": dict(num_batches=1024, batch_size=1, num_draws=128),
    "utah": dict(num_batches=64, batch_size=16, num_draws=128),
}

ROOT = pathlib.Path(__file__).resolve().parents[1]


def data_path(name):
    return ROOT / "data" / f"noaa_{name}_tmax.npz"


def weights_path(name):
    return ROOT / "weights" / f"mambaflownp_{name}.npz"


def load_model(name, weights=None):
    """The released model of a region."""
    model = MambaFlowNP(MODELS[name], rngs=nnx.Rngs(0))
    return load_checkpoint(model, weights or weights_path(name))


class Region:
    """Prepared region file. See the Data section of the README for its fields."""

    def __init__(self, path):
        with np.load(path) as f:
            self.station_ids = f["station_ids"]
            self.lonlat = f["lonlat"]
            self.coords = f["coords"]
            self.values = f["tmax"]
            self.dates = f["dates"]
            self.split = {"train": f["train_idx"], "val": f["val_idx"], "test": f["test_idx"]}
            self.normalizer = QuantileNormalizer(
                f["qt_raw"], f["qt_z"], f["qt_z_unique"], f["qt_raw_unique"])

    def sampler(self, split, seed):
        """Returns a FieldSampler over normalized values of one split."""
        z = self.normalizer.transform(self.values[self.split[split]])
        return FieldSampler(self.coords, z, seed=seed)
