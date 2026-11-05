# MambaFlowNP

Code and pretrained weights for MambaFlowNP (NeurIPS 2026).

MambaFlowNP is a neural process for irregularly observed spatial fields with linear-time
inference. Points are ordered along jittered Hilbert curves and processed by distance-aware
Mamba-2 layers. The model is trained with flow matching, distilled into a one-step sampler with
reflow, and fine-tuned on CRPS.

## Installation

The data files are stored with Git LFS. Run `git lfs install` before cloning.

```bash
pip install -r requirements.txt
pip install "jax[cuda12]==0.9.0.1"  # NVIDIA GPU
```

Tested with Python 3.14.

## Repository layout

| Path | |
|---|---|
| `mambaflownp/` | Model, sampling, training and data code |
| `train.py` | Training |
| `evaluate.py` | Evaluation |
| `reproduce.sh` | Evaluates the released weights on Utah, California and US and renders the figures |
| `weights/` | Weights for Utah, California and US |
| `data/` | NOAA daily maximum temperature for Utah, California and US |
| `figures.py`, `plotting/` | Figures in `results/` |
| `results/` | Evaluation results and figures |
| `tests/` | Tests |

## Inference

```python
import jax

import mambaflownp as mf
from mambaflownp import noaa

region = noaa.Region(noaa.data_path("california"))
model = noaa.load_model("california")

# 243 context and 243 query stations on one test day, with values in normalised units.
x_ctx, y_ctx, x_qry, y_qry = region.sampler("test", seed=0).sample(1, 243, 243)
keys = jax.random.split(jax.random.PRNGKey(0), 128)
samples = region.normalizer.inverse(mf.sample(model, x_ctx, y_ctx, x_qry, keys)[..., 0])
# samples: (128, 243) in °C
```

The same keys give the same samples on any set of queries, whole or in chunks.

To map your own inputs: `mf.CoordFrame.fit(region.lonlat).transform(lonlat)` for coordinates,
`region.normalizer.transform` for values in °C.

## Evaluation

```bash
./reproduce.sh
```

10 seeds × 1024 test tasks × 128 samples per task (16 samples from each of 8 scan-grid shifts).
Context and target points are split equally: 111 each for Utah, 243 for California, 2790 for
the US. Map samples are cached in `results/qualitative/`.

## Training

```bash
python train.py --region utah --out runs/utah
python train.py --region california --out runs/california
python train.py --region us --out runs/us
```

| | Utah | California | US |
|---|---|---|---|
| Hidden size | 128 | 256 | 256 |
| Layers | 6 | 6 | 6 |
| State size | 16 | 16 | 16 |
| Fourier frequencies | 256 | 192 | 384 |
| Fourier scale | 0.25 | 0.25 | 0.25 |
| Scan-grid shifts | 8 | 8 | 8 |
| Points per task | 222 | 486 | 5580 |
| Batch size | 16 | 16 | 8 |
| Learning rate | 1.0e-3 | 1.52e-3 | 1.0e-3 |
| Flow-matching steps | 80,000 | 80,000 | 80,000 |
| Reflow couplings | 5,000 | 5,000 | 1,000 |
| Reflow steps | 20,000 | 20,000 | 20,000 |
| CRPS batch size | 8 | 8 | 2 |
| CRPS samples per task | 8 | 8 | 4 |

Optimizer and schedule settings are in `mambaflownp/training.py`.

## Data

NOAA GHCN-Daily (<https://www.ncei.noaa.gov/pub/data/ghcn/daily/>) daily maximum temperature.
Training: 2005 to 2016. Validation: 2017 to 2019. Test: 2020 to 2024.

| Region | Stations | Train / validation / test days |
|---|---|---|
| Utah | 255 | 4383 / 1095 / 1827 |
| California | 523 | 4383 / 1095 / 1827 |
| US | 5943 | 4383 / 1095 / 1827 |

Each file contains `station_ids`, `lonlat`, `coords` (unit square), `tmax` (°C, NaN if missing),
`dates`, `train_idx`, `val_idx`, `test_idx`, and the quantile transform `qt_raw`, `qt_z`,
`qt_z_unique` and `qt_raw_unique`.

## Results

Test set, mean ± standard error over 10 seeds.

| Region | RMSE (°C) | CRPS (°C) | 90% coverage | 95% coverage |
|---|---|---|---|---|
| Utah | 1.570 ± 0.005 | 0.767 ± 0.002 | 0.903 ± 0.000 | 0.940 ± 0.000 |
| California | 1.344 ± 0.002 | 0.681 ± 0.001 | 0.911 ± 0.000 | 0.945 ± 0.000 |
| US | 1.598 ± 0.003 | 0.774 ± 0.001 | 0.926 ± 0.002 | 0.955 ± 0.001 |

Baseline results are listed in `results/baselines.csv`.

| Model | Paper | Repo | Citation |
|---|---|---|---|
| CNP | [Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html) | [google-deepmind/neural-processes](https://github.com/google-deepmind/neural-processes) | Garnelo et al., ICML 2018 |
| NP | [Neural Processes](https://arxiv.org/abs/1807.01622) | [google-deepmind/neural-processes](https://github.com/google-deepmind/neural-processes) | Garnelo et al., ICML Workshop 2018 |
| CANP | [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) | [google-deepmind/neural-processes](https://github.com/google-deepmind/neural-processes) | Kim et al., ICLR 2019 |
| ANP | [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) | [google-deepmind/neural-processes](https://github.com/google-deepmind/neural-processes) | Kim et al., ICLR 2019 |
| TNPA | [Transformer Neural Processes: Uncertainty-Aware Meta Learning Via Sequence Modeling](https://proceedings.mlr.press/v162/nguyen22b.html) | [tung-nd/TNP-pytorch](https://github.com/tung-nd/TNP-pytorch) | Nguyen and Grover, ICML 2022 |
| NDP | [Neural Diffusion Processes](https://proceedings.mlr.press/v202/dutordoir23a.html) | [vdutor/neural-diffusion-processes](https://github.com/vdutor/neural-diffusion-processes) | Dutordoir et al., ICML 2023 |
| FlowNP | [Flow Matching Neural Processes](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a92519f525c00085095fa41c5c46cdb5-Abstract-Conference.html) | [danrsm/flowNP](https://github.com/danrsm/flowNP) | Abu Hamad and Rosenbaum, NeurIPS 2025 |

![Utah](results/pareto_noaa_utah.png)

![California](results/pareto_noaa_california.png)

![US](results/pareto_noaa_us.png)

![Utah, 25% context](results/noaa_qualitative_utah.png)

![California, 25% context](results/noaa_qualitative_california.png)

![US, 10% context](results/noaa_qualitative_us.png)

### California heat wave, September 2022

MambaFlowNP maps on test days, with half of the stations as context.

![California heat wave, September 2022](results/heatwave_california.gif)

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests
```

## Citation

```bibtex
@inproceedings{santoni2026mambaflownp,
  title     = {Mamba Flow Matching Neural Processes: Linear-Time Inference for Irregularly
               Observed Spatial Fields},
  author    = {Santoni, Cosmo and Charles, Giovanni and Hitge, Timothy James and
               Watson, Oliver John and Semenova, Elizaveta},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Apache License 2.0. See `LICENSE`.
