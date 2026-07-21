#!/usr/bin/env bash
set -euo pipefail

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
NUM_CLASSES="${NUM_CLASSES:-10}"
EVAL_GAP="${EVAL_GAP:-1}"
COMMON_EVAL_BATCH_SIZE="${COMMON_EVAL_BATCH_SIZE:-256}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-id,ood,mix}"
DATASETS_CSV="${DATASETS:-Cifar10,FashionMNIST}"
ALGORITHMS_CSV="${ALGORITHMS:-FedAvg,FedProx,FedCross,FedBN,FedALA,FedAS,pFedMe,cwFedAvg,FedDST,PMOE_FedPer,FedCP,DualFed}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
STREAM_LOGS="${STREAM_LOGS:-false}"
AUTO_SUMMARIZE="${AUTO_SUMMARIZE:-true}"
SUMMARY_TARGET_RUNS="${SUMMARY_TARGET_RUNS:-1}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$SCRIPT_DIR/fedprism_idoodmix_result.csv}"

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

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: ${PYTHON_BIN:-<empty>}. Set FEDCD_PYTHON." >&2
  exit 2
fi
for required_path in "$SYSTEM_DIR" "$FL_DATA_ROOT"; do
  if [[ ! -d "$required_path" ]]; then
    echo "[ERROR] Required directory not found: $required_path" >&2
    exit 2
  fi
done
if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL_JOBS" -lt 1 ]]; then
  echo "[ERROR] MAX_PARALLEL_JOBS must be a positive integer: $MAX_PARALLEL_JOBS" >&2
  exit 2
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

dataset_name_for() {
  local dataset_base="${1,,}"
  case "$dataset_base" in
    cifar10|cifar-10)
      printf 'Cifar10_fedprism_idoodmix_nc%s' "$NUM_CLIENTS"
      ;;
    fashionmnist|fashion-mnist)
      printf 'FashionMNIST_fedprism_idoodmix_nc%s' "$NUM_CLIENTS"
      ;;
    *)
      echo "[ERROR] Unsupported dataset: $1" >&2
      return 2
      ;;
  esac
}

IFS=',' read -r -a datasets <<< "$DATASETS_CSV"
IFS=',' read -r -a algorithms <<< "$ALGORITHMS_CSV"
IFS=',' read -r -a scenarios <<< "$EVAL_SCENARIOS"

for scenario_raw in "${scenarios[@]}"; do
  scenario="$(trim "$scenario_raw")"
  case "$scenario" in
    id|ood|mix) ;;
    *)
      echo "[ERROR] EVAL_SCENARIOS accepts only id,ood,mix: $scenario" >&2
      exit 2
      ;;
  esac
done

validate_dataset() {
  local dataset="$1"
  "$PYTHON_BIN" - "$FL_DATA_ROOT/$dataset" "$dataset" "$NUM_CLIENTS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_name = sys.argv[2]
num_clients = int(sys.argv[3])
config_path = root / 'config.json'
if not config_path.is_file():
    raise SystemExit(f'missing config.json: {config_path}')
with config_path.open(encoding='utf-8') as handle:
    config = json.load(handle)
expected_source = 'Cifar10' if dataset_name.startswith('Cifar10_') else 'FashionMNIST'
expected = {
    'schema': 'fedprism_id_ood_mix_v1',
    'dataset': expected_source,
    'num_clients': num_clients,
    'num_classes': 10,
}
errors = [
    f'{key}={config.get(key)!r} (expected {value!r})'
    for key, value in expected.items()
    if config.get(key) != value
]
for split in ('train', 'test/id', 'test/ood', 'test/mix'):
    split_dir = root / split
    actual_ids = sorted(
        int(path.stem) for path in split_dir.glob('*.npz') if path.stem.isdigit()
    ) if split_dir.is_dir() else []
    if actual_ids != list(range(num_clients)):
        errors.append(f'{split} client files are incomplete')
if not (root / 'test' / 'pool.npz').is_file():
    errors.append('test/pool.npz is missing')
if errors:
    raise SystemExit('; '.join(errors))
PY
}

for dataset_base_raw in "${datasets[@]}"; do
  dataset_base="$(trim "$dataset_base_raw")"
  dataset="$(dataset_name_for "$dataset_base")"
  validate_dataset "$dataset"
done

date_str="$(date -u +%Y%m%d)"
time_str="$(date -u +%H%M%S)"
model_tag="${MODEL,,}"
run_tag="fedprism_idoodmix_${model_tag}_${date_str}_${time_str}_pid$$"
queue_root="$SCRIPT_DIR/batch_runs/fedprism_idoodmix_${model_tag}/date_${date_str}/time_${time_str}_pid$$"
run_log_dir="$queue_root/run_logs"
mkdir -p "$run_log_dir"
status_csv="$queue_root/status.csv"
printf '%s\n' 'idx,total,dataset_base,algorithm,eval_scenarios,num_clients,seed,dataset,status,exit_code,start_utc,end_utc,run_log' > "$status_csv"

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
  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local status='ok'
  if [[ "$exit_code" -ne 0 ]]; then
    status='failed'
    fail_count=$((fail_count + 1))
    echo "[FAIL] $label exit_code=$exit_code"
    tail -n 40 "$run_log" || true
  else
    echo "[DONE] $label"
  fi
  local row_idx row_total row_dataset row_algorithm row_scenarios
  local row_num_clients row_seed row_dataset_name
  IFS='|' read -r row_idx row_total row_dataset row_algorithm row_scenarios \
    row_num_clients row_seed row_dataset_name <<< "$label"
  printf '%s\n' "${row_idx},${row_total},${row_dataset},${row_algorithm},${row_scenarios},${row_num_clients},${row_seed},${row_dataset_name},${status},${exit_code},${start_utc},${end_utc},${run_log}" >> "$status_csv"
  pids=("${pids[@]:1}")
  labels=("${labels[@]:1}")
  logs=("${logs[@]:1}")
  starts=("${starts[@]:1}")
}

launch_job() {
  local dataset_base="$1"
  local algorithm="$2"
  local dataset
  dataset="$(dataset_name_for "$dataset_base")"
  local run_log="$run_log_dir/$(printf '%03d' "$idx")_${dataset_base}_${algorithm}_seed${SEED}.log"
  local goal="${algorithm}_${model_tag}_idoodmix_nc${NUM_CLIENTS}_${run_tag}_seed${SEED}"
  local start_utc
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local extra_args=()

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
  echo "[CONFIG] scenarios=$EVAL_SCENARIOS rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO nc=$NUM_CLIENTS"
  echo "[LOG] $run_log"
  echo "=========================================================="

  local command_args=(
    "$PYTHON_BIN" -u main.py
    -data "$dataset"
    -ncl "$NUM_CLASSES"
    -m "$MODEL"
    -algo "$algorithm"
    -gr "$GLOBAL_ROUNDS"
    -lr "$LR"
    -lbs "$LBS"
    -ls "$LOCAL_EPOCHS"
    -nc "$NUM_CLIENTS"
    -jr "$JOIN_RATIO"
    -t "$TIMES"
    --seed "$SEED"
    -eg "$EVAL_GAP"
    --common_eval_batch_size "$COMMON_EVAL_BATCH_SIZE"
    --eval_common_global False
    --eval-scenarios "$EVAL_SCENARIOS"
    -go "$goal"
    -dev "$DEVICE"
    -did "$DEVICE_ID"
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
  labels+=("${idx}|${total}|${dataset_base}|${algorithm}|${EVAL_SCENARIOS}|${NUM_CLIENTS}|${SEED}|${dataset}")
  logs+=("$run_log")
  starts+=("$start_utc")
}

echo "[INFO] FedPRISM ID/OOD/Mix queue root: $queue_root"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[INFO] Datasets: $DATASETS_CSV"
echo "[INFO] Algorithms: $ALGORITHMS_CSV"
echo "[INFO] Eval scenarios: $EVAL_SCENARIOS"
echo "[INFO] Max parallel jobs: $MAX_PARALLEL_JOBS"

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

if [[ "${AUTO_SUMMARIZE,,}" =~ ^(1|true|yes|y|on)$ ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/tools/summarize_fedprism_scenarios.py" \
    --logs-root "$SCRIPT_DIR/logs" \
    --datasets "$DATASETS_CSV" \
    --methods "$ALGORITHMS_CSV" \
    --model "$MODEL" \
    --clients "$NUM_CLIENTS" \
    --scenarios "$EVAL_SCENARIOS" \
    --required-round "$((GLOBAL_ROUNDS + 1))" \
    --target-runs "$SUMMARY_TARGET_RUNS" \
    --scale percent \
    --decimals 2 \
    --output-csv "$SUMMARY_OUTPUT"
  echo "[INFO] Scenario summary: $SUMMARY_OUTPUT"
fi
