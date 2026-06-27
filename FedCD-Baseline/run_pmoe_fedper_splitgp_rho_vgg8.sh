#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/home1/irteam/.conda/envs/pfllib/bin/python}"
FL_DATA_ROOT="${FL_DATA_ROOT:-/home1/irteam/workspace/fl_data}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FL_DATA_ROOT

LOCAL_PYTHON_BIN="/home1/irteam/.conda/envs/pfllib/bin/python"
if [[ ! -x "$PYTHON_BIN" && -x "$LOCAL_PYTHON_BIN" ]]; then
  echo "[WARN] Python interpreter not found: $PYTHON_BIN" >&2
  echo "[WARN] Falling back to: $LOCAL_PYTHON_BIN" >&2
  PYTHON_BIN="$LOCAL_PYTHON_BIN"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
if [[ -d "$PYTHON_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib:${LD_LIBRARY_PATH:-}"
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
PMOE_TOPK="${PMOE_TOPK:-8}"
PMOE_FINETUNE_EPOCHS="${PMOE_FINETUNE_EPOCHS:-50}"
PMOE_LR="${PMOE_LR:-0.5}"
PMOE_LOCK_EXPERTS="${PMOE_LOCK_EXPERTS:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${MAX_PARALLEL_RHOS:-1}}"
DATASETS_CSV="${DATASETS:-Cifar10,FashionMNIST}"
RHOS_CSV="${RHOS:-0.0,0.2,0.4,0.6,0.8}"
EVAL_RHOS="${EVAL_RHOS:-0.0,0.2,0.4,0.6,0.8}"

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL_JOBS" -lt 1 ]]; then
  echo "[ERROR] MAX_PARALLEL_JOBS must be a positive integer: $MAX_PARALLEL_JOBS" >&2
  exit 1
fi

date_str="$(date -u +%Y%m%d)"
time_str="$(date -u +%H%M%S)"
run_tag="pmoe_fedper_splitgp_rho_${date_str}_${time_str}"
queue_root="$SCRIPT_DIR/batch_runs/pmoe_fedper_splitgp_rho_vgg8/date_${date_str}/time_${time_str}"
run_log_dir="$queue_root/run_logs"
mkdir -p "$run_log_dir"
status_csv="$queue_root/status.csv"
printf '%s\n' 'idx,total,dataset_base,algorithm,rho,num_clients,seed,dataset,status,exit_code,start_utc,end_utc,run_log' > "$status_csv"

IFS=',' read -r -a datasets <<< "$DATASETS_CSV"
IFS=',' read -r -a rhos <<< "$RHOS_CSV"
total=$(( ${#datasets[@]} * ${#rhos[@]} ))
idx=0

for dataset_base_check in "${datasets[@]}"; do
  dataset_base_check="$(echo "$dataset_base_check" | xargs)"
  for rho_check in "${rhos[@]}"; do
    rho_check="$(echo "$rho_check" | xargs)"
    dataset_check="${dataset_base_check}_splitgp_pat_rho${rho_check}_nc${NUM_CLIENTS}"
    if [[ ! -d "$FL_DATA_ROOT/$dataset_check" ]]; then
      echo "[ERROR] Missing dataset: $FL_DATA_ROOT/$dataset_check" >&2
      exit 1
    fi
  done
done

cd "$SYSTEM_DIR" || exit 1

eval_rho_args=(--eval-rhos "$EVAL_RHOS")

echo "[INFO] PMOE_FedPer SplitGP rho queue root: $queue_root"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[INFO] PMOE_FedPer topk=$PMOE_TOPK, fine_tuning_epochs=$PMOE_FINETUNE_EPOCHS, moe_lr=$PMOE_LR, lock_experts=$PMOE_LOCK_EXPERTS"
echo "[INFO] Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "[INFO] Eval rhos: $EVAL_RHOS"

batch_pids=()
batch_idxs=()
batch_dataset_bases=()
batch_rhos=()
batch_datasets=()
batch_logs=()
batch_starts=()

cleanup_children() {
  if [[ ${#batch_pids[@]} -gt 0 ]]; then
    echo "[INFO] Stopping ${#batch_pids[@]} running rho job(s)..." >&2
    kill "${batch_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup_children INT TERM

wait_for_batch() {
  local i pid exit_code status end_utc
  for i in "${!batch_pids[@]}"; do
    pid="${batch_pids[$i]}"
    if wait "$pid"; then
      exit_code=0
    else
      exit_code=$?
    fi
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ $exit_code -eq 0 ]]; then
      status="ok"
      echo "[DONE ${batch_idxs[$i]}/$total] dataset=${batch_datasets[$i]} algo=PMOE_FedPer seed=$SEED"
    else
      status="failed"
      echo "[FAIL ${batch_idxs[$i]}/$total] dataset=${batch_datasets[$i]} algo=PMOE_FedPer seed=$SEED exit_code=$exit_code"
      tail -n 40 "${batch_logs[$i]}" || true
    fi
    printf '%s\n' "${batch_idxs[$i]},${total},${batch_dataset_bases[$i]},PMOE_FedPer,${batch_rhos[$i]},${NUM_CLIENTS},${SEED},${batch_datasets[$i]},${status},${exit_code},${batch_starts[$i]},${end_utc},${batch_logs[$i]}" >> "$status_csv"
    echo
  done
  batch_pids=()
  batch_idxs=()
  batch_dataset_bases=()
  batch_rhos=()
  batch_datasets=()
  batch_logs=()
  batch_starts=()
}

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
    goal="PMOE_FedPer_splitgp_rho${rho}_nc${NUM_CLIENTS}_${run_tag}_seed${SEED}"
    run_log="$run_log_dir/${idx}_${dataset_base}_PMOE_FedPer_rho${safe_rho}_nc${NUM_CLIENTS}_seed${SEED}.log"
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    echo "=========================================================="
    echo "[START $idx/$total] dataset=$dataset algo=PMOE_FedPer seed=$SEED"
    echo "[LOG] $run_log"
    echo "=========================================================="

    "$PYTHON_BIN" -u main.py \
      -data "$dataset" \
      -ncl "$num_classes" \
      -m "$MODEL" \
      -algo "PMOE_FedPer" \
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
      "${eval_rho_args[@]}" \
      -tk "$PMOE_TOPK" \
      -mfte "$PMOE_FINETUNE_EPOCHS" \
      -moelr "$PMOE_LR" \
      -le "$PMOE_LOCK_EXPERTS" > "$run_log" 2>&1 &
    pid=$!

    batch_pids+=("$pid")
    batch_idxs+=("$idx")
    batch_dataset_bases+=("$dataset_base")
    batch_rhos+=("$rho")
    batch_datasets+=("$dataset")
    batch_logs+=("$run_log")
    batch_starts+=("$start_utc")

    if [[ ${#batch_pids[@]} -ge $MAX_PARALLEL_JOBS ]]; then
      wait_for_batch
    fi
  done
done

if [[ ${#batch_pids[@]} -gt 0 ]]; then
  wait_for_batch
fi

echo "[INFO] PMOE_FedPer SplitGP rho queue finished."
echo "[INFO] Status CSV: $status_csv"

