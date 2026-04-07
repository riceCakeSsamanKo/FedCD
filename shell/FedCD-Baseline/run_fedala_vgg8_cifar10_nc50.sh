#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="0"
GLOBAL_ROUNDS="100"
LR="0.005"
LBS="128"
LOCAL_EPOCHS="2"
JOIN_RATIO="1.0"
NUM_CLASSES="10"
TIMES="1"
ETA="1.0"
RAND_PERCENT="80"
LAYER_IDX="2"

scenarios=("pat_nc50" "dir0.1_nc50" "dir0.5_nc50" "dir1.0_nc50")

cd "$SYSTEM_DIR"

for scenario in "${scenarios[@]}"; do
    dataset="Cifar10_${scenario}"
    goal="FedALA_${scenario}"
    echo "=========================================================="
    echo "[START] algo=FedALA scenario=$scenario dataset=$dataset"
    echo "=========================================================="

    "$PYTHON_BIN" -u main.py \
        -data "$dataset" \
        -ncl "$NUM_CLASSES" \
        -m "$MODEL" \
        -algo FedALA \
        -gr "$GLOBAL_ROUNDS" \
        -lr "$LR" \
        -lbs "$LBS" \
        -ls "$LOCAL_EPOCHS" \
        -nc 50 \
        -jr "$JOIN_RATIO" \
        -t "$TIMES" \
        -go "$goal" \
        -dev "$DEVICE" \
        -did "$DEVICE_ID" \
        -et "$ETA" \
        -s "$RAND_PERCENT" \
        -p "$LAYER_IDX"

    echo "[DONE] algo=FedALA scenario=$scenario"
    echo
done
