#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="/home/mulsoap0504/FedCD/FedCD-Core/dataset"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
FL_DATA_ROOT="${FL_DATA_ROOT:-/home/mulsoap0504/FedCD/fl_data}"
RAW_ROOT="${RAW_ROOT:-$FL_DATA_ROOT/FashionMNIST/rawdata}"

mkdir -p "$FL_DATA_ROOT"

run_generate() {
  local alpha="$1"
  local dataset_name="$2"
  echo "=========================================================="
  echo "[FashionMNIST] generating $dataset_name"
  echo "[CONFIG] partition=dir num_clients=50 alpha=$alpha raw_root=$RAW_ROOT"
  echo "=========================================================="
  (
    cd "$GEN_DIR"
    "$PYTHON_BIN" generate_FashionMNIST.py noniid - dir 50 "$alpha" "$FL_DATA_ROOT/$dataset_name" "$RAW_ROOT"
  )
  echo
}

run_generate 0.1 FashionMNIST_dir0.1_nc50
run_generate 0.5 FashionMNIST_dir0.5_nc50
run_generate 1.0 FashionMNIST_dir1.0_nc50

echo "[INFO] FashionMNIST NC50 DIR fl_data generation complete."
