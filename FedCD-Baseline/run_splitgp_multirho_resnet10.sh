#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-$(command -v python)}"
FL_DATA_ROOT="${FL_DATA_ROOT:-/home1/irteam/workspace/fl_data}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FL_DATA_ROOT
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: ${PYTHON_BIN:-<empty>}. Set FEDCD_PYTHON." >&2
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

MODEL="${MODEL:-ResNet10}"
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
NUM_CLASSES="${NUM_CLASSES:-10}"
EVAL_GAP="${EVAL_GAP:-1}"
COMMON_EVAL_BATCH_SIZE="${COMMON_EVAL_BATCH_SIZE:-256}"
TRAIN_RHO="${TRAIN_RHO:-0.0}"
EVAL_RHOS="${EVAL_RHOS:-0.0,0.2,0.4,0.6,0.8}"
DATASET_NAME_OVERRIDE="${DATASET_NAME_OVERRIDE:-}"
ENABLE_MULTI_RHO_EVAL="${ENABLE_MULTI_RHO_EVAL:-true}"
DATASETS_CSV="${DATASETS:-Cifar10,FashionMNIST}"
ALGORITHMS_CSV="${ALGORITHMS:-FedAvg,FedProx,FedCross,FedBN,FedALA,FedAS,pFedMe,cwFedAvg,FedDST,PMOE_FedPer,FedCP,DualFed}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
STREAM_LOGS="${STREAM_LOGS:-false}"
DYNAMIC_CLIENT_ENABLED="${DYNAMIC_CLIENT_ENABLED:-false}"
DYNAMIC_CLIENT_JOIN_ROUND="${DYNAMIC_CLIENT_JOIN_ROUND:-51}"
DYNAMIC_CLIENT_OLD_CLASSES="${DYNAMIC_CLIENT_OLD_CLASSES:-0,1,2,3,4,5}"
DYNAMIC_CLIENT_NEW_CLASSES="${DYNAMIC_CLIENT_NEW_CLASSES:-6,7,8,9}"

FEDDST_SPARSITY="${FEDDST_SPARSITY:-0.3}"
FEDDST_FINAL_SPARSITY="${FEDDST_FINAL_SPARSITY:-$FEDDST_SPARSITY}"
FEDDST_READJUSTMENT_RATIO="${FEDDST_READJUSTMENT_RATIO:-0.5}"
FEDDST_READJUSTMENT_INTERVAL="${FEDDST_READJUSTMENT_INTERVAL:-10}"
FEDDST_SPARSITY_DISTRIBUTION="${FEDDST_SPARSITY_DISTRIBUTION:-erk}"
FEDDST_RATE_DECAY_METHOD="${FEDDST_RATE_DECAY_METHOD:-cosine}"
PMOE_TOPK="${PMOE_TOPK:-8}"
PMOE_FINETUNE_EPOCHS="${PMOE_FINETUNE_EPOCHS:-50}"
PMOE_LR="${PMOE_LR:-0.5}"
PMOE_LOCK_EXPERTS="${PMOE_LOCK_EXPERTS:-0}"
FEDCP_LAMDA="${FEDCP_LAMDA:-1.0}"
DUALFED_CON_LAMBDA="${DUALFED_CON_LAMBDA:-0.1}"
DUALFED_CON_TEMP="${DUALFED_CON_TEMP:-0.5}"

if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL_JOBS" -lt 1 ]]; then
  echo "[ERROR] MAX_PARALLEL_JOBS must be a positive integer: $MAX_PARALLEL_JOBS" >&2
  exit 1
fi

IFS=',' read -r -a datasets <<< "$DATASETS_CSV"
IFS=',' read -r -a algorithms <<< "$ALGORITHMS_CSV"
IFS=',' read -r -a eval_rho_array <<< "$EVAL_RHOS"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

for dataset_base_raw in "${datasets[@]}"; do
  dataset_base="$(trim "$dataset_base_raw")"
  if [[ -n "$DATASET_NAME_OVERRIDE" ]]; then
    train_dataset="$DATASET_NAME_OVERRIDE"
  else
    train_dataset="${dataset_base}_splitgp_pat_rho${TRAIN_RHO}_nc${NUM_CLIENTS}"
  fi
  if [[ ! -d "$FL_DATA_ROOT/$train_dataset" ]]; then
    echo "[ERROR] Missing train dataset: $FL_DATA_ROOT/$train_dataset" >&2
    exit 1
  fi
  if [[ "${ENABLE_MULTI_RHO_EVAL,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    for rho_raw in "${eval_rho_array[@]}"; do
      rho="$(trim "$rho_raw")"
      eval_dataset="${dataset_base}_splitgp_pat_rho${rho}_nc${NUM_CLIENTS}"
      if [[ ! -d "$FL_DATA_ROOT/$eval_dataset" ]]; then
        echo "[ERROR] Missing eval dataset: $FL_DATA_ROOT/$eval_dataset" >&2
        exit 1
      fi
    done
  fi
done

date_str="$(date -u +%Y%m%d)"
time_str="$(date -u +%H%M%S)"
model_tag="${MODEL,,}"
mode_tag="standard"
if [[ "${DYNAMIC_CLIENT_ENABLED,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  mode_tag="dynamic_clients"
fi
run_tag="splitgp_multirho_${model_tag}_${mode_tag}_trainrho${TRAIN_RHO}_${date_str}_${time_str}_pid$$"
queue_root="$SCRIPT_DIR/batch_runs/splitgp_multirho_${model_tag}_${mode_tag}/date_${date_str}/time_${time_str}_pid$$"
run_log_dir="$queue_root/run_logs"
mkdir -p "$run_log_dir"
status_csv="$queue_root/status.csv"
printf '%s\n' 'idx,total,dataset_base,algorithm,train_rho,eval_rhos,num_clients,seed,dynamic_clients,dynamic_join_round,dataset,status,exit_code,start_utc,end_utc,run_log' > "$status_csv"

total=$(( ${#datasets[@]} * ${#algorithms[@]} ))
idx=0
fail_count=0
pids=()
labels=()
logs=()
starts=()

finish_first_job() {
  local pid="${pids[0]}"
  local label="${labels[0]}"
  local run_log="${logs[0]}"
  local start_utc="${starts[0]}"
  local exit_code=0
  if wait "$pid"; then
    exit_code=0
  else
    exit_code=$?
  fi
  local end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local status="ok"
  if [[ "$exit_code" -ne 0 ]]; then
    status="failed"
    fail_count=$((fail_count + 1))
    echo "[FAIL] $label exit_code=$exit_code"
    tail -n 40 "$run_log" || true
  else
    echo "[DONE] $label"
  fi
  IFS='|' read -r row_idx row_total row_dataset row_algorithm row_train_rho row_eval_rhos row_num_clients row_seed row_dynamic row_join_round row_dataset_name <<< "$label"
  printf '%s\n' "${row_idx},${row_total},${row_dataset},${row_algorithm},${row_train_rho},${row_eval_rhos},${row_num_clients},${row_seed},${row_dynamic},${row_join_round},${row_dataset_name},${status},${exit_code},${start_utc},${end_utc},${run_log}" >> "$status_csv"
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
  logs=("${logs[@]:1}")
  starts=("${starts[@]:1}")
}

launch_job() {
  local dataset_base="$1"
  local algorithm="$2"
  local dataset="${dataset_base}_splitgp_pat_rho${TRAIN_RHO}_nc${NUM_CLIENTS}"
  if [[ -n "$DATASET_NAME_OVERRIDE" ]]; then
    dataset="$DATASET_NAME_OVERRIDE"
  fi
  local safe_train_rho="${TRAIN_RHO//./p}"
  local run_log="$run_log_dir/$(printf '%03d' "$idx")_${dataset_base}_${algorithm}_trainrho${safe_train_rho}_seed${SEED}.log"
  local goal="${algorithm}_${model_tag}_trainrho${TRAIN_RHO}_multirho_nc${NUM_CLIENTS}_${run_tag}_seed${SEED}"
  local start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local extra_args=()
  local dynamic_args=()
  local eval_args=()
  if [[ "${ENABLE_MULTI_RHO_EVAL,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    eval_args+=("--eval-rhos" "$EVAL_RHOS")
  fi
  if [[ "${DYNAMIC_CLIENT_ENABLED,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    dynamic_args+=(
      "--dynamic_client_enabled" "True"
      "--dynamic_client_join_round" "$DYNAMIC_CLIENT_JOIN_ROUND"
      "--dynamic_client_old_classes" "$DYNAMIC_CLIENT_OLD_CLASSES"
      "--dynamic_client_new_classes" "$DYNAMIC_CLIENT_NEW_CLASSES"
      "--dynamic_client_expected_existing_clients" "30"
      "--dynamic_client_expected_newcomer_clients" "20"
      "--dynamic_client_require_contiguous_ids" "True"
      "--eval_common_global" "False"
    )
  fi
  case "$algorithm" in
    FedProx)
      extra_args+=("-mu" "1.0")
      ;;
    FedALA)
      extra_args+=("-et" "1.0" "-s" "80" "-p" "2")
      ;;
    FedCross)
      extra_args+=("-fsb" "0" "-ca" "0.99" "-cmss" "1")
      ;;
    cwFedAvg)
      extra_args+=("-cw" "-wdr" "-plt" "-ncw" "1" "-wd" "10")
      ;;
    FedDST)
      extra_args+=(
        "--feddst_sparsity" "$FEDDST_SPARSITY"
        "--feddst_final_sparsity" "$FEDDST_FINAL_SPARSITY"
        "--feddst_readjustment_ratio" "$FEDDST_READJUSTMENT_RATIO"
        "--feddst_rounds_between_readjustments" "$FEDDST_READJUSTMENT_INTERVAL"
        "--feddst_sparsity_distribution" "$FEDDST_SPARSITY_DISTRIBUTION"
        "--feddst_rate_decay_method" "$FEDDST_RATE_DECAY_METHOD"
      )
      ;;
    PMOE_FedPer)
      extra_args+=("-tk" "$PMOE_TOPK" "-mfte" "$PMOE_FINETUNE_EPOCHS" "-moelr" "$PMOE_LR" "-le" "$PMOE_LOCK_EXPERTS")
      ;;
    FedCP)
      extra_args+=("-lam" "$FEDCP_LAMDA")
      ;;
    DualFed)
      extra_args+=("--dualfed_con_lambda" "$DUALFED_CON_LAMBDA" "--dualfed_con_temp" "$DUALFED_CON_TEMP")
      ;;
  esac

  echo "=========================================================="
  echo "[LAUNCH $idx/$total] dataset=$dataset algorithm=$algorithm model=$MODEL seed=$SEED"
  echo "[CONFIG] train_rho=$TRAIN_RHO eval_rhos=$EVAL_RHOS rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO nc=$NUM_CLIENTS"
  echo "[LOG] $run_log"
  echo "=========================================================="

  local command_args=(
      "$PYTHON_BIN" -u main.py
      -data "$dataset" \
      -ncl "$NUM_CLASSES" \
      -m "$MODEL" \
      -algo "$algorithm" \
      -gr "$GLOBAL_ROUNDS" \
      -lr "$LR" \
      -lbs "$LBS" \
      -ls "$LOCAL_EPOCHS" \
      -nc "$NUM_CLIENTS" \
      -jr "$JOIN_RATIO" \
      -t "$TIMES" \
      --seed "$SEED" \
      -eg "$EVAL_GAP" \
      --common_eval_batch_size "$COMMON_EVAL_BATCH_SIZE" \
      --fedprism_eval_match True \
      --fedprism_eval_reserved_fraction 0.2 \
      --fedprism_eval_reserved_seed 0 \
      -go "$goal" \
      -dev "$DEVICE" \
      -did "$DEVICE_ID" \
      "${eval_args[@]}" \
      "${dynamic_args[@]}" \
      "${extra_args[@]}"
  )

  (
    cd "$SYSTEM_DIR" || exit 1
    if [[ "${STREAM_LOGS,,}" =~ ^(1|true|yes|y|on)$ ]]; then
      set -o pipefail
      "${command_args[@]}" 2>&1 | tee "$run_log"
      exit "${PIPESTATUS[0]}"
    fi
    "${command_args[@]}" > "$run_log" 2>&1
  ) &

  pids+=("$!")
  labels+=("${idx}|${total}|${dataset_base}|${algorithm}|${TRAIN_RHO}|${EVAL_RHOS}|${NUM_CLIENTS}|${SEED}|${DYNAMIC_CLIENT_ENABLED}|${DYNAMIC_CLIENT_JOIN_ROUND}|${dataset}")
  logs+=("$run_log")
  starts+=("$start_utc")
}

echo "[INFO] $MODEL SplitGP train-once multi-rho queue root: $queue_root"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[INFO] Datasets: $DATASETS_CSV"
echo "[INFO] Algorithms: $ALGORITHMS_CSV"
echo "[INFO] Train rho: $TRAIN_RHO"
echo "[INFO] Eval rhos: $EVAL_RHOS"
echo "[INFO] Dynamic clients: $DYNAMIC_CLIENT_ENABLED (join round: $DYNAMIC_CLIENT_JOIN_ROUND)"
echo "[INFO] Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "[INFO] Stream logs: $STREAM_LOGS"
echo ""

for dataset_base_raw in "${datasets[@]}"; do
  dataset_base="$(trim "$dataset_base_raw")"
  for algorithm_raw in "${algorithms[@]}"; do
    algorithm="$(trim "$algorithm_raw")"
    idx=$((idx + 1))
    while [[ "${#pids[@]}" -ge "$MAX_PARALLEL_JOBS" ]]; do
      finish_first_job
    done
    launch_job "$dataset_base" "$algorithm"
  done
done

while [[ "${#pids[@]}" -gt 0 ]]; do
  finish_first_job
done

echo "[INFO] Finished. Status CSV: $status_csv"
if [[ "$fail_count" -gt 0 ]]; then
  echo "[WARN] Failed jobs: $fail_count" >&2
  exit 1
fi
exit 0
