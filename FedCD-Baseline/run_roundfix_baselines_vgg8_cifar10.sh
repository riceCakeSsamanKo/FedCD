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
DEFAULT_QUEUE_ROOT="$SCRIPT_DIR/batch_runs/roundfix_vgg8_cifar10/date_${DATE_STR}/time_${TIME_STR}"
STATUS_CSV="${ROUND_FIX_STATUS_CSV:-$DEFAULT_QUEUE_ROOT/status.csv}"
QUEUE_ROOT="$(dirname "$STATUS_CSV")"
mkdir -p "$QUEUE_ROOT"

if [[ ! -f "$STATUS_CSV" ]]; then
    {
        echo "algorithm,scenario,dataset,status,exit_code,start_utc,end_utc"
    } > "$STATUS_CSV"
fi

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

run_one() {
    local algo="$1"
    local scenario="$2"
    local num_clients="$3"
    local dataset="Cifar10_${scenario}"
    local goal="${algo}_${scenario}_roundfix_${DATE_STR}_${TIME_STR}"
    local start_utc
    local end_utc
    local exit_code=0
    local status="ok"
    local -a extra_args=()

    case "$algo" in
        FedProx)
            extra_args+=(-mu 1.0)
            ;;
        FedKD)
            extra_args+=(-mlr 0.005 -Ts 0.95 -Te 0.98)
            ;;
        Ditto)
            extra_args+=(-mu 1.0 -pls 1)
            ;;
        FedNTD)
            extra_args+=(-bt 1.0 -tau 1.0)
            ;;
    esac

    if rg -n "^${algo},${scenario},${dataset},ok,0," "$STATUS_CSV" >/dev/null 2>&1; then
        echo "[SKIP] algo=$algo scenario=$scenario already completed"
        echo
        return 0
    fi

    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "=========================================================="
    echo "[START] algo=$algo scenario=$scenario dataset=$dataset"
    echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO"
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        echo "[EXTRA] ${extra_args[*]}"
    fi
    echo "=========================================================="

    "$PYTHON_BIN" -u "$SYSTEM_DIR/main.py" \
        -data "$dataset" \
        -ncl "$NUM_CLASSES" \
        -m "$MODEL" \
        -algo "$algo" \
        -gr "$GLOBAL_ROUNDS" \
        -lr "$LR" \
        -lbs "$LBS" \
        -ls "$LOCAL_EPOCHS" \
        -nc "$num_clients" \
        -jr "$JOIN_RATIO" \
        -t "$TIMES" \
        -go "$goal" \
        -dev "$DEVICE" \
        -did "$DEVICE_ID" \
        "${extra_args[@]}"
    exit_code=$?

    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ $exit_code -ne 0 ]]; then
        status="failed"
        echo "[FAIL] algo=$algo scenario=$scenario exit_code=$exit_code"
    else
        echo "[DONE] algo=$algo scenario=$scenario"
    fi

    {
        echo "${algo},${scenario},${dataset},${status},${exit_code},${start_utc},${end_utc}"
    } >> "$STATUS_CSV"

    echo
}

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Using python: $PYTHON_BIN"
echo

# Fix all corrupted Ditto CIFAR-10 NC20 runs.
for scenario in pat_nc20 dir0.1_nc20 dir0.5_nc20 dir1.0_nc20; do
    run_one "Ditto" "$scenario" 20
done

# Fix all corrupted Ditto CIFAR-10 NC50 runs.
for scenario in pat_nc50 dir0.1_nc50 dir0.5_nc50 dir1.0_nc50; do
    run_one "Ditto" "$scenario" 50
done

# Resume the interrupted NC50 tail after Ditto.
for algo in FedBN FedNTD; do
    for scenario in pat_nc50 dir0.1_nc50 dir0.5_nc50 dir1.0_nc50; do
        run_one "$algo" "$scenario" 50
    done
done

echo "[INFO] Round-fix baseline queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
