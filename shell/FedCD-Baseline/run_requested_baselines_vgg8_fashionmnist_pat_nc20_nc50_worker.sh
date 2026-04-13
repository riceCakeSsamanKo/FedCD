#!/bin/bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <gpu_id> <worker_tag> <algo1> [algo2 ...]" >&2
    exit 1
fi

GPU_ID="$1"
shift
WORKER_TAG="$1"
shift
algorithms=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

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
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/baselines_vgg8_fashionmnist_pat_nc20_nc50_parallel/${WORKER_TAG}/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"
STATUS_CSV="$QUEUE_ROOT/status.csv"
printf '%s\n' 'algorithm,scenario,dataset,status,exit_code,start_utc,end_utc,gpu,worker' > "$STATUS_CSV"

MODEL="VGG8"
DEVICE="cuda"
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

scenarios=("pat_nc20" "pat_nc50")

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Using python: $PYTHON_BIN"
echo "[INFO] GPU: $GPU_ID"
echo "[INFO] Worker: $WORKER_TAG"
echo "[INFO] Algorithms: ${algorithms[*]}"
echo "[INFO] fl_data root: /home/mulsoap0504/FedCD/fl_data"
echo

for algo in "${algorithms[@]}"; do
  for scenario in "${scenarios[@]}"; do
    if [[ "$scenario" == "pat_nc20" ]]; then
      dataset="FashionMNIST_pat_nc20"
      num_clients="20"
    else
      dataset="FashionMNIST_pat_nc50"
      num_clients="50"
    fi
    goal="${algo}_${scenario}_${DATE_STR}_${TIME_STR}_${WORKER_TAG}"
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit_code=0
    extra_args=()

    case "$algo" in
      FedProx)
        extra_args+=(-mu 1.0)
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
    echo "[START] algo=$algo scenario=$scenario dataset=$dataset gpu=$GPU_ID worker=$WORKER_TAG"
    echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO nc=$num_clients"
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
      -nc "$num_clients" \
      -jr "$JOIN_RATIO" \
      -t "$TIMES" \
      -go "$goal" \
      -dev "$DEVICE" \
      -did "$GPU_ID" \
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
    printf '%s\n' "${algo},${scenario},${dataset},${status},${exit_code},${start_utc},${end_utc},${GPU_ID},${WORKER_TAG}" >> "$STATUS_CSV"
    echo
  done
done

echo "[INFO] Worker queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
