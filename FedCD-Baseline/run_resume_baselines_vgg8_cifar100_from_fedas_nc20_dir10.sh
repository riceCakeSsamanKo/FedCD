#!/bin/bash
set -euo pipefail

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
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/baselines_vgg8_cifar100_resume_from_fedas_nc20_dir10/date_${DATE_STR}/time_${TIME_STR}"
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
NUM_CLASSES="100"
TIMES="1"

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Resume queue root: $QUEUE_ROOT"
echo "[INFO] Resuming CIFAR-100 baselines from FedAS NC20 dir1.0"
echo

run_item() {
    local algo="$1"
    local scenario="$2"
    local nc="$3"
    local dataset="Cifar100_${scenario}"
    local goal="${algo}_${scenario}_cifar100_resume_${DATE_STR}_${TIME_STR}"
    local start_utc exit_code end_utc status
    local extra_args=()

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

    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit_code=0

    echo "=========================================================="
    echo "[START] algo=$algo scenario=$scenario dataset=$dataset"
    echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO nc=$nc"
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
        echo "[DONE] algo=$algo scenario=$scenario"
    else
        status="failed"
        echo "[FAIL] algo=$algo scenario=$scenario exit_code=$exit_code"
    fi

    {
        echo "${algo},${scenario},${dataset},${status},${exit_code},${start_utc},${end_utc}"
    } >> "$STATUS_CSV"

    echo
}

# Resume NC20 from the first missing point: FedAS dir1.0
run_item "FedAS" "dir1.0_nc20" 20

# Continue the remaining NC20 algorithms in the original order.
for algo in FedKD FedAvg FedProx LG-FedAvg Ditto FedBN FedNTD; do
    for scenario in pat_nc20 dir0.1_nc20 dir0.5_nc20 dir1.0_nc20; do
        run_item "$algo" "$scenario" 20
    done
done

echo "[INFO] NC20 resume portion finished. Continuing with full NC50 queue."
bash "$SCRIPT_DIR/run_requested_baselines_vgg8_cifar100_nc50.sh"

echo "[INFO] CIFAR-100 resume queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
