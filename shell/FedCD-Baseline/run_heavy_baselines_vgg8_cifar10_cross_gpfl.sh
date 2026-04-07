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
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/heavy_cross_gpfl_cifar10/date_${DATE_STR}/time_${TIME_STR}"
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

algorithms=("FedCross" "GPFL")
num_clients_list=(20 50)
scenarios=("pat" "dir0.1" "dir0.5" "dir1.0")

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Running FedCross + GPFL for CIFAR-10"

auto_num_classes() {
    local dataset="$1"
    if [[ "$dataset" == Cifar10_* ]]; then echo 10; else echo 100; fi
}

for algo in "${algorithms[@]}"; do
    for nc in "${num_clients_list[@]}"; do
        for scenario in "${scenarios[@]}"; do
            dataset="Cifar10_${scenario}_nc${nc}"
            num_classes="$(auto_num_classes "$dataset")"
            goal="${algo}_${scenario}_nc${nc}_heavy_${DATE_STR}_${TIME_STR}"
            start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            exit_code=0
            extra_args=()

            echo "=========================================================="
            echo "[START] algo=$algo dataset=$dataset"
            echo "=========================================================="

            "$PYTHON_BIN" -u main.py \
                -data "$dataset" \
                -ncl "$num_classes" \
                -m "$MODEL" \
                -algo "$algo" \
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
                "${extra_args[@]}"
            exit_code=$?

            end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            if [[ $exit_code -eq 0 ]]; then
                status="ok"
                echo "[DONE] algo=$algo dataset=$dataset"
            else
                status="failed"
                echo "[FAIL] algo=$algo dataset=$dataset exit_code=$exit_code"
            fi
            echo "${algo},${scenario}_nc${nc},${dataset},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
            echo
        done
    done
done

echo "[INFO] Finished heavy CIFAR-10 queue."
echo "[INFO] Status CSV: $STATUS_CSV"
