#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/baselines_vgg8_mnist_custom_nc50/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"
STATUS_CSV="$QUEUE_ROOT/status.csv"
printf '%s\n' 'algorithm,scenario,dataset,status,exit_code,start_utc,end_utc' > "$STATUS_CSV"

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

algorithms=("FedAS" "FedKD" "FedAvg" "FedProx" "Ditto" "FedBN" "FedALA" "FedCross" "cwFedAvg")
scenarios=("pat_nc50" "dir0.1_nc50" "dir0.5_nc50" "dir1.0_nc50")

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Using python: $PYTHON_BIN"
echo "[INFO] fl_data root: /home/mulsoap0504/FedCD/fl_data"
echo

for algo in "${algorithms[@]}"; do
  for scenario in "${scenarios[@]}"; do
    dataset="MNIST_${scenario}"
    goal="${algo}_${scenario}_${DATE_STR}_${TIME_STR}"
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit_code=0
    extra_args=()
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
      FedALA)
        extra_args+=(-et "$ETA" -s "$RAND_PERCENT" -p "$LAYER_IDX")
        ;;
      cwFedAvg)
        extra_args+=(-cw -wdr -plt -ncw 1 -wd 10)
        ;;
    esac

    echo "=========================================================="
    echo "[START] algo=$algo scenario=$scenario dataset=$dataset"
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
      -nc 50 \
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
    printf '%s\n' "${algo},${scenario},${dataset},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
    echo
  done
done

echo "[INFO] All requested MNIST NC50 custom baseline queue items finished."
echo "[INFO] Status CSV: $STATUS_CSV"
