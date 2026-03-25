#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python interpreter not found: $PYTHON_BIN"
    exit 1
fi

if [[ ! -d "$SYSTEM_DIR" ]]; then
    echo "[ERROR] System directory not found: $SYSTEM_DIR"
    exit 1
fi

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="$(date -u +%H%M%S)"
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/lgfedavg_vgg8_cifar10_all/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"

STATUS_CSV="$QUEUE_ROOT/status.csv"
{
    echo "algorithm,scenario,dataset,status,exit_code,start_utc,end_utc"
} > "$STATUS_CSV"

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
ALGO="LG-FedAvg"
scenarios=("pat_nc20" "dir0.1_nc20" "dir0.5_nc20" "dir1.0_nc20" "pat_nc50" "dir0.1_nc50" "dir0.5_nc50" "dir1.0_nc50")

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Using python: $PYTHON_BIN"
echo "[INFO] Starting LG-FedAvg on CIFAR-10 (NC20/50, pat/dir0.1/0.5/1.0)"
echo

for scenario in "${scenarios[@]}"; do
    dataset="Cifar10_${scenario}"
    if [[ $scenario =~ nc([0-9]+) ]]; then
        nc="${BASH_REMATCH[1]}"
    else
        nc="20"
    fi
    goal="${ALGO}_${scenario}_${DATE_STR}_${TIME_STR}"
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "=========================================================="
    echo "[START] algo=$ALGO scenario=$scenario dataset=$dataset"
    echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO"
    echo "=========================================================="

    "$PYTHON_BIN" -u main.py \
        -data "$dataset" \
        -ncl "$NUM_CLASSES" \
        -m "$MODEL" \
        -algo "$ALGO" \
        -gr "$GLOBAL_ROUNDS" \
        -lr "$LR" \
        -lbs "$LBS" \
        -ls "$LOCAL_EPOCHS" \
        -nc "$nc" \
        -jr "$JOIN_RATIO" \
        -t "$TIMES" \
        -go "$goal" \
        -dev "$DEVICE" \
        -did "$DEVICE_ID"
    exit_code=$?

    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ $exit_code -eq 0 ]]; then
        status="ok"
        echo "[DONE] algo=$ALGO scenario=$scenario"
    else
        status="failed"
        echo "[FAIL] algo=$ALGO scenario=$scenario exit_code=$exit_code"
    fi

    echo "${ALGO},${scenario},${dataset},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
    echo

done

echo "[INFO] LG-FedAvg CIFAR-10 queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
