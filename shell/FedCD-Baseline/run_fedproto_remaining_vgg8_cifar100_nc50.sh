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
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/fedproto_remaining_cifar100_nc50/date_${DATE_STR}/time_${TIME_STR}"
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
NUM_CLASSES="100"
TIMES="1"
LAMDA="1.0"
NC="50"
scenarios=("dir0.5_nc50" "dir1.0_nc50")

is_complete() {
    local partition="$1"
    local alpha="$2"
    local path="$SCRIPT_DIR/logs/Cifar100/FedProto/GM_VGG8/$partition"
    if [[ "$partition" == "dir" ]]; then
        path="$path/$alpha"
    fi
    path="$path/NC_${NC}"
    if [[ ! -d "$path" ]]; then
        return 1
    fi
    while IFS= read -r acc; do
        local last_round
        last_round="$(tail -n 1 "$acc" | cut -d, -f1)"
        if [[ "$last_round" =~ ^[0-9]+$ ]] && [[ "$last_round" -ge 101 ]]; then
            return 0
        fi
    done < <(find "$path" -name acc.csv | sort)
    return 1
}

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Running remaining FedProto for CIFAR-100 NC50"

for scenario in "${scenarios[@]}"; do
    partition="dir"
    alpha="${scenario#dir}"
    alpha="${alpha%_nc50}"

    dataset="Cifar100_${scenario}"
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if is_complete "$partition" "$alpha"; then
        echo "[SKIP] FedProto $dataset already has a completed 101-round log"
        end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "FedProto,${scenario},${dataset},skipped_complete,0,${start_utc},${end_utc}" >> "$STATUS_CSV"
        echo
        continue
    fi

    goal="FedProto_${scenario}_remaining_${DATE_STR}_${TIME_STR}"
    exit_code=0

    echo "=========================================================="
    echo "[START] algo=FedProto dataset=$dataset"
    echo "=========================================================="

    "$PYTHON_BIN" -u main.py \
        -data "$dataset" \
        -ncl "$NUM_CLASSES" \
        -m "$MODEL" \
        -algo FedProto \
        -gr "$GLOBAL_ROUNDS" \
        -lr "$LR" \
        -lbs "$LBS" \
        -ls "$LOCAL_EPOCHS" \
        -nc "$NC" \
        -jr "$JOIN_RATIO" \
        -t "$TIMES" \
        -go "$goal" \
        -dev "$DEVICE" \
        -did "$DEVICE_ID" \
        -lam "$LAMDA"
    exit_code=$?

    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ $exit_code -eq 0 ]]; then
        status="ok"
        echo "[DONE] algo=FedProto dataset=$dataset"
    else
        status="failed"
        echo "[FAIL] algo=FedProto dataset=$dataset exit_code=$exit_code"
    fi
    echo "FedProto,${scenario},${dataset},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
    echo

done

echo "[INFO] Finished remaining FedProto CIFAR-100 NC50 queue."
echo "[INFO] Status CSV: $STATUS_CSV"
