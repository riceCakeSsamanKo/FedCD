#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="/home/mulsoap0504/FedCD/FedCD-Core/dataset"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
FL_DATA_ROOT="${FL_DATA_ROOT:-/home/mulsoap0504/FedCD/fl_data}"
RAW_ROOT="${RAW_ROOT:-$FL_DATA_ROOT/MNIST/rawdata}"

mkdir -p "$FL_DATA_ROOT"

run_generate() {
  local partition="$1"
  local num_clients="$2"
  local alpha="$3"
  local dataset_name="$4"
  echo "=========================================================="
  echo "[MNIST] generating $dataset_name"
  echo "[CONFIG] partition=$partition num_clients=$num_clients alpha=$alpha raw_root=$RAW_ROOT"
  echo "=========================================================="
  (
    cd "$GEN_DIR"
    "$PYTHON_BIN" generate_MNIST.py noniid - "$partition" "$num_clients" "$alpha" "$FL_DATA_ROOT/$dataset_name" "$RAW_ROOT"
  )
  echo
}

run_generate pat 20 0.5 MNIST_pat_nc20
run_generate dir 20 0.1 MNIST_dir0.1_nc20
run_generate dir 20 0.5 MNIST_dir0.5_nc20
run_generate dir 20 1.0 MNIST_dir1.0_nc20

run_generate pat 50 0.5 MNIST_pat_nc50
run_generate dir 50 0.1 MNIST_dir0.1_nc50
run_generate dir 50 0.5 MNIST_dir0.5_nc50
run_generate dir 50 1.0 MNIST_dir1.0_nc50

echo "[INFO] MNIST fl_data generation complete."
