#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/ds1home/aislab/miniconda3/envs/pfllib/bin/python}"
FL_DATA_ROOT="${FL_DATA_ROOT:-/ds1home/aislab/Min/data/fl_data}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FL_DATA_ROOT

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$SYSTEM_DIR" ]]; then
  echo "[ERROR] System directory not found: $SYSTEM_DIR" >&2
  exit 1
fi
if [[ ! -d "$FL_DATA_ROOT" ]]; then
  echo "[ERROR] FL data root not found: $FL_DATA_ROOT" >&2
  exit 1
fi

MODEL="${MODEL:-VGG8}"
DEVICE="${DEVICE:-cuda}"
DEVICE_ID="${DEVICE_ID:-0}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
LR="${LR:-0.005}"
LBS="${LBS:-128}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
JOIN_RATIO="${JOIN_RATIO:-1.0}"
TIMES="${TIMES:-1}"
SEED="${SEED:-1}"
NUM_CLIENTS="${NUM_CLIENTS:-50}"
FEDDST_SPARSITY="${FEDDST_SPARSITY:-0.3}"
FEDDST_FINAL_SPARSITY="${FEDDST_FINAL_SPARSITY:-$FEDDST_SPARSITY}"
FEDDST_READJUSTMENT_RATIO="${FEDDST_READJUSTMENT_RATIO:-0.5}"
FEDDST_READJUSTMENT_INTERVAL="${FEDDST_READJUSTMENT_INTERVAL:-10}"
FEDDST_SPARSITY_DISTRIBUTION="${FEDDST_SPARSITY_DISTRIBUTION:-erk}"
FEDDST_RATE_DECAY_METHOD="${FEDDST_RATE_DECAY_METHOD:-cosine}"
DATASETS_CSV="${DATASETS:-Cifar10,FashionMNIST}"
RHOS_CSV="${RHOS:-0.0,0.2,0.4,0.6,0.8}"

date_str="$(date -u +%Y%m%d)"
time_str="$(date -u +%H%M%S)"
run_tag="feddst_splitgp_rho_${date_str}_${time_str}"
queue_root="$SCRIPT_DIR/batch_runs/feddst_splitgp_rho_vgg8/date_${date_str}/time_${time_str}"
run_log_dir="$queue_root/run_logs"
mkdir -p "$run_log_dir"
status_csv="$queue_root/status.csv"
printf '%s\n' 'idx,total,dataset_base,algorithm,rho,num_clients,seed,dataset,status,exit_code,start_utc,end_utc,run_log' > "$status_csv"

IFS=',' read -r -a datasets <<< "$DATASETS_CSV"
IFS=',' read -r -a rhos <<< "$RHOS_CSV"
total=$(( ${#datasets[@]} * ${#rhos[@]} ))
idx=0

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] FedDST SplitGP rho queue root: $queue_root"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[INFO] FedDST sparsity=$FEDDST_SPARSITY readjustment_ratio=$FEDDST_READJUSTMENT_RATIO interval=$FEDDST_READJUSTMENT_INTERVAL"

for dataset_base in "${datasets[@]}"; do
  dataset_base="$(echo "$dataset_base" | xargs)"
  num_classes="10"
  for rho in "${rhos[@]}"; do
    rho="$(echo "$rho" | xargs)"
    dataset="${dataset_base}_splitgp_pat_rho${rho}_nc${NUM_CLIENTS}"
    if [[ ! -d "$FL_DATA_ROOT/$dataset" ]]; then
      echo "[ERROR] Missing dataset: $FL_DATA_ROOT/$dataset" >&2
      exit 1
    fi

    idx=$((idx + 1))
    safe_rho="${rho//./p}"
    goal="FedDST_splitgp_rho${rho}_nc${NUM_CLIENTS}_${run_tag}_seed${SEED}"
    run_log="$run_log_dir/${idx}_${dataset_base}_FedDST_rho${safe_rho}_nc${NUM_CLIENTS}_seed${SEED}.log"
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "=========================================================="
    echo "[START $idx/$total] dataset=$dataset algo=FedDST seed=$SEED"
    echo "[LOG] $run_log"
    echo "=========================================================="

    "$PYTHON_BIN" -u main.py \
      -data "$dataset" \
      -ncl "$num_classes" \
      -m "$MODEL" \
      -algo "FedDST" \
      -gr "$GLOBAL_ROUNDS" \
      -lr "$LR" \
      -lbs "$LBS" \
      -ls "$LOCAL_EPOCHS" \
      -nc "$NUM_CLIENTS" \
      -jr "$JOIN_RATIO" \
      -t "$TIMES" \
      --seed "$SEED" \
      -go "$goal" \
      -dev "$DEVICE" \
      -did "$DEVICE_ID" \
      --feddst_sparsity "$FEDDST_SPARSITY" \
      --feddst_final_sparsity "$FEDDST_FINAL_SPARSITY" \
      --feddst_readjustment_ratio "$FEDDST_READJUSTMENT_RATIO" \
      --feddst_rounds_between_readjustments "$FEDDST_READJUSTMENT_INTERVAL" \
      --feddst_sparsity_distribution "$FEDDST_SPARSITY_DISTRIBUTION" \
      --feddst_rate_decay_method "$FEDDST_RATE_DECAY_METHOD" > "$run_log" 2>&1
    exit_code=$?
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if [[ $exit_code -eq 0 ]]; then
      status="ok"
      echo "[DONE $idx/$total] dataset=$dataset algo=FedDST seed=$SEED"
    else
      status="failed"
      echo "[FAIL $idx/$total] dataset=$dataset algo=FedDST seed=$SEED exit_code=$exit_code"
      tail -n 40 "$run_log" || true
    fi
    printf '%s\n' "${idx},${total},${dataset_base},FedDST,${rho},${NUM_CLIENTS},${SEED},${dataset},${status},${exit_code},${start_utc},${end_utc},${run_log}" >> "$status_csv"
    echo
  done
done

echo "[INFO] FedDST SplitGP rho queue finished."
echo "[INFO] Status CSV: $status_csv"

