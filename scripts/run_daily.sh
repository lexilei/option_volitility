#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

statarb download-data --config configs/universe.yaml --output data/processed/prices.parquet
statarb generate-weights --config configs/live.yaml --prices data/processed/prices.parquet
statarb trade-paper --config configs/live.yaml --prices data/processed/prices.parquet
