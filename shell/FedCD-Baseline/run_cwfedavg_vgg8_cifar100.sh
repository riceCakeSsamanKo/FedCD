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

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="$(date -u +%H%M%S)"
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/cwfedavg_cifar100/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"
STATUS_CSV="$QUEUE_ROOT/status.csv"
echo "algorithm,scenario,dataset,status,exit_code,start_utc,end_utc" > "$STATUS_CSV"

MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="0"
GLOBAL_ROUNDS="100"
LR="0.005"
LBS="128"
LOCAL_EPOCHS="2"
JOIN_RATIO="1.0"
TIMES="1"

num_clients_list=(20 50)
scenarios=("pat" "dir0.1" "dir0.5" "dir1.0")

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Running cwFedAvg for CIFAR-100"

for nc in "${num_clients_list[@]}"; do
    for scenario in "${scenarios[@]}"; do
        dataset="Cifar100_${scenario}_nc${nc}"
        goal="cwFedAvg_${scenario}_nc${nc}_${DATE_STR}_${TIME_STR}"
        start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        exit_code=0

        echo "=========================================================="
        echo "[START] algo=cwFedAvg dataset=$dataset"
        echo "=========================================================="

        "$PYTHON_BIN" -u main.py \
            -data "$dataset" \
            -ncl 100 \
            -m "$MODEL" \
            -algo cwFedAvg \
            -gr "$GLOBAL_ROUNDS" \
            -lr "$LR" \
            -lbs "$LBS" \
            -ls "$LOCAL_EPOCHS" \
            -nc "$nc" \
            -jr "$JOIN_RATIO" \
            -t "$TIMES" \
            -go "$goal" \
            -dev "$DEVICE" \
            -did "$DEVICE_ID" \
            -cw -wdr -plt -ncw 1 -wd 10
        exit_code=$?

        end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if [[ $exit_code -eq 0 ]]; then
            status="ok"
            echo "[DONE] algo=cwFedAvg dataset=$dataset"
        else
            status="failed"
            echo "[FAIL] algo=cwFedAvg dataset=$dataset exit_code=$exit_code"
        fi
        echo "cwFedAvg,${scenario}_nc${nc},${dataset},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
        echo
    done
done

echo "[INFO] Finished cwFedAvg CIFAR-100 queue."
echo "[INFO] Status CSV: $STATUS_CSV"
