#!/bin/bash
# Evaluates the released weights on Utah, California and US and renders the figures.
set -euo pipefail
cd "$(dirname "$0")"
python evaluate.py --region utah --out results/utah
python evaluate.py --region california --out results/california
python evaluate.py --region us --out results/us --draws-per-call 16
python figures.py
