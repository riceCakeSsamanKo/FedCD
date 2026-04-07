#!/bin/bash

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <status_csv_path>"
    exit 1
fi

STATUS_INPUT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

if [[ ! -f "$STATUS_INPUT" ]]; then
    echo "[ERROR] Status CSV not found: $STATUS_INPUT"
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python interpreter not found: $PYTHON_BIN"
    exit 1
fi

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="$(date -u +%H%M%S)"
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/baselines_vgg8_cifar10_nc20_rerun/date_${DATE_STR}/time_${TIME_STR}"
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

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Rerun source status CSV: $STATUS_INPUT"
echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Using python: $PYTHON_BIN"
echo

tail -n +2 "$STATUS_INPUT" | while IFS=, read -r algo scenario dataset status exit_code _rest; do
    if [[ "$status" != "failed" ]]; then
        continue
    fi

    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_exit_code=0
    extra_args=()

    case "$algo" in
        Ditto)
            extra_args+=(-mu 1.0 -pls 1)
            ;;
        FedNTD)
            extra_args+=(-bt 1.0 -tau 1.0)
            ;;
    esac

    goal="${algo}_${scenario}_retry_${DATE_STR}_${TIME_STR}"

    echo "=========================================================="
    echo "[RERUN] algo=$algo scenario=$scenario dataset=$dataset"
    echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO"
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        echo "[EXTRA] ${extra_args[*]}"
    fi
    echo "=========================================================="

    "$PYTHON_BIN" -u main.py \
        -data "$dataset" \
        -ncl "$NUM_CLASSES" \
        -m "$MODEL" \
        -algo "$algo" \
        -gr "$GLOBAL_ROUNDS" \
        -lr "$LR" \
        -lbs "$LBS" \
        -ls "$LOCAL_EPOCHS" \
        -nc 20 \
        -jr "$JOIN_RATIO" \
        -t "$TIMES" \
        -go "$goal" \
        -dev "$DEVICE" \
        -did "$DEVICE_ID" \
        "${extra_args[@]}"
    run_exit_code=$?

    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ $run_exit_code -eq 0 ]]; then
        run_status="ok"
        echo "[DONE] algo=$algo scenario=$scenario"
    else
        run_status="failed"
        echo "[FAIL] algo=$algo scenario=$scenario exit_code=$run_exit_code"
    fi

    {
        echo "${algo},${scenario},${dataset},${run_status},${run_exit_code},${start_utc},${end_utc}"
    } >> "$STATUS_CSV"

    echo
done

echo "[INFO] Failed baseline reruns finished."
echo "[INFO] Status CSV: $STATUS_CSV"
