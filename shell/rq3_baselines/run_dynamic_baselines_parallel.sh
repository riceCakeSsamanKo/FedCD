#!/usr/bin/env bash
set -uo pipefail

# Run the RQ3 dynamic-client baselines in deterministic method batches.
# One active method always means two concurrent experiments:
#   1) CIFAR-10 and 2) FashionMNIST.

usage() {
  cat <<'EOF'
Usage:
  bash shell/rq3_baselines/run_dynamic_baselines_parallel.sh [N_METHODS] [SEED] [METHODS]

Arguments:
  N_METHODS  Maximum number of methods active at once (default: 2).
             The maximum number of concurrent experiments is 2 * N_METHODS.
  SEED       Baseline seed (default: 1).
  METHODS    Optional comma-separated method list. This takes precedence over
             METHODS_CSV (example: FedAvg,FedProx,FedBN).

Environment variables:
  BASELINE_DIR             FedCD-Baseline directory. By default, use the
                           sibling FedCD/FedCD-Baseline checkout.
  FL_DATA_ROOT             FL data root. By default, use the sibling fl_data.
  FEDCD_PYTHON             Python interpreter (default: command -v python).
  GPU_IDS                  Comma/space-separated physical GPU IDs. If unset,
                           detect all GPUs with nvidia-smi; fall back to 0.
  METHODS_CSV              Optional comma-separated method list override.
  LIVE_LOGS                Stream per-job logs to this terminal while retaining
                           log files (default: 1; set to 0 to disable).
  LAUNCH_STAGGER_SECONDS   Delay between wrapper starts (default: 2). This
                           avoids same-second queue metadata collisions.

Examples:
  # FedAvg and FedProx are the first two active methods. Their two datasets
  # produce four concurrent experiments, assigned round-robin to GPUs 0-3.
  GPU_IDS=0,1,2,3 bash shell/rq3_baselines/run_dynamic_baselines_parallel.sh 2 1

  # Run at most three methods (six experiments) concurrently on GPUs 0 and 1.
  GPU_IDS=0,1 bash shell/rq3_baselines/run_dynamic_baselines_parallel.sh 3 1

  # Run only the selected methods and show their logs live.
  GPU_IDS=0 bash shell/rq3_baselines/run_dynamic_baselines_parallel.sh \
    2 1 FedAvg,FedProx,FedBN
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MAX_PARALLEL_METHODS="${1:-${MAX_PARALLEL_METHODS:-2}}"
RUN_SEED="${2:-${SEED:-1}}"
METHODS_ARG="${3:-}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-2}"
LIVE_LOGS="${LIVE_LOGS:-1}"
EXPERIMENT_MODEL="${MODEL:-VGG8}"
EXPERIMENT_GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
EXPERIMENT_LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
EXPERIMENT_JOIN_RATIO="${JOIN_RATIO:-1.0}"
EXPERIMENT_LR="${LR:-0.005}"
EXPERIMENT_BATCH_SIZE="${LBS:-128}"
EXPERIMENT_NUM_CLIENTS="${NUM_CLIENTS:-50}"

if ! [[ "$MAX_PARALLEL_METHODS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] N_METHODS must be a positive integer: $MAX_PARALLEL_METHODS" >&2
  exit 2
fi
if ! [[ "$RUN_SEED" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SEED must be a non-negative integer: $RUN_SEED" >&2
  exit 2
fi
if ! [[ "$LAUNCH_STAGGER_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] LAUNCH_STAGGER_SECONDS must be a non-negative number." >&2
  exit 2
fi
if ! [[ "${LIVE_LOGS,,}" =~ ^(1|true|yes|y|on|0|false|no|n|off)$ ]]; then
  echo "[ERROR] LIVE_LOGS must be a boolean value: $LIVE_LOGS" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDCCM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$FEDCCM_DIR/.." && pwd)"

BASELINE_DIR="${BASELINE_DIR:-$WORKSPACE_DIR/FedCD/FedCD-Baseline}"
FL_DATA_ROOT="${FL_DATA_ROOT:-$WORKSPACE_DIR/fl_data}"
FEDCD_PYTHON_BIN="${FEDCD_PYTHON:-$(command -v python 2>/dev/null || true)}"

DEFAULT_METHODS=(
  FedAvg
  FedProx
  FedCross
  FedBN
  FedALA
  FedAS
  pFedMe
  cwFedAvg
  FedDST
  FedCP
  DualFed
)

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

METHODS=()
METHODS_INPUT="${METHODS_ARG:-${METHODS_CSV:-}}"
if [[ -n "$METHODS_INPUT" ]]; then
  IFS=',' read -r -a method_tokens <<< "$METHODS_INPUT"
  for token in "${method_tokens[@]}"; do
    method="$(trim "$token")"
    [[ -n "$method" ]] && METHODS+=("$method")
  done
else
  METHODS=("${DEFAULT_METHODS[@]}")
fi

if [[ "${#METHODS[@]}" -eq 0 ]]; then
  echo "[ERROR] No methods were selected." >&2
  exit 2
fi

if [[ ! -d "$BASELINE_DIR" ]]; then
  echo "[ERROR] FedCD-Baseline directory not found: $BASELINE_DIR" >&2
  echo "        Set BASELINE_DIR to the Linux server checkout." >&2
  exit 2
fi
if [[ ! -x "$FEDCD_PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found or not executable: $FEDCD_PYTHON_BIN" >&2
  exit 2
fi

CIFAR_SCRIPT="$BASELINE_DIR/dynamic_clients/run_cifar10.sh"
FMNIST_SCRIPT="$BASELINE_DIR/dynamic_clients/run_fashionmnist.sh"
for required_path in \
  "$CIFAR_SCRIPT" \
  "$FMNIST_SCRIPT" \
  "$FL_DATA_ROOT/Cifar10_dynamic_clients_nc50" \
  "$FL_DATA_ROOT/FashionMNIST_dynamic_clients_nc50"
do
  if [[ ! -e "$required_path" ]]; then
    echo "[ERROR] Required RQ3 path not found: $required_path" >&2
    exit 2
  fi
done

GPU_ID_LIST=()
if [[ -n "${GPU_IDS:-}" ]]; then
  gpu_text="${GPU_IDS//,/ }"
  read -r -a gpu_tokens <<< "$gpu_text"
  for token in "${gpu_tokens[@]}"; do
    gpu="$(trim "$token")"
    [[ -n "$gpu" ]] && GPU_ID_LIST+=("$gpu")
  done
elif command -v nvidia-smi >/dev/null 2>&1; then
  while IFS= read -r gpu; do
    gpu="$(trim "$gpu")"
    [[ -n "$gpu" ]] && GPU_ID_LIST+=("$gpu")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
fi

if [[ "${#GPU_ID_LIST[@]}" -eq 0 ]]; then
  GPU_ID_LIST=(0)
fi

RUN_DATE="$(date -u +%Y%m%d)"
RUN_TIME="$(date -u +%H%M%S)"
RUN_DIR="$BASELINE_DIR/batch_runs/rq3_dynamic_parallel/date_${RUN_DATE}/time_${RUN_TIME}_pid$$"
WRAPPER_LOG_DIR="$RUN_DIR/wrapper_logs"
MPL_ROOT="$RUN_DIR/mpl"
STATUS_TSV="$RUN_DIR/status.tsv"
mkdir -p "$WRAPPER_LOG_DIR" "$MPL_ROOT"
printf '%s\n' $'batch\tmethod\tdataset\tseed\tgpu\tstatus\texit_code\tstart_utc\tend_utc\twrapper_log' > "$STATUS_TSV"

ACTIVE_PIDS=()
on_interrupt() {
  echo "[INTERRUPT] Terminating active RQ3 wrappers..." >&2
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit 130
}
trap on_interrupt INT TERM

LAST_PID=""
LAST_LOG=""
LAST_START=""

launch_dataset() {
  local batch_number="$1"
  local method="$2"
  local dataset_key="$3"
  local gpu_id="$4"
  local dataset_script
  local log_prefix

  case "$dataset_key" in
    cifar10)
      dataset_script="$CIFAR_SCRIPT"
      ;;
    fashionmnist)
      dataset_script="$FMNIST_SCRIPT"
      ;;
    *)
      echo "[ERROR] Unknown dataset key: $dataset_key" >&2
      return 2
      ;;
  esac

  LAST_LOG="$WRAPPER_LOG_DIR/batch${batch_number}_${method}_${dataset_key}_seed${RUN_SEED}.log"
  LAST_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log_prefix="[$method/$dataset_key/gpu$gpu_id]"
  mkdir -p "$MPL_ROOT/${method}_${dataset_key}_seed${RUN_SEED}"

  if [[ "${LIVE_LOGS,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    (
      set -o pipefail
      (
        cd "$BASELINE_DIR" || exit 2
        export FEDCD_PYTHON="$FEDCD_PYTHON_BIN"
        export FL_DATA_ROOT
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to logical 0.
        export DEVICE_ID=0
        export MPLCONFIGDIR="$MPL_ROOT/${method}_${dataset_key}_seed${RUN_SEED}"
        export MAX_PARALLEL_JOBS=1
        export SEED="$RUN_SEED"
        export ALGORITHMS="$method"
        export STREAM_LOGS=1
        bash "$dataset_script"
      ) 2>&1 | tee "$LAST_LOG" | sed -u "s|^|$log_prefix |"
      exit "${PIPESTATUS[0]}"
    ) &
  else
    (
      cd "$BASELINE_DIR" || exit 2
      export FEDCD_PYTHON="$FEDCD_PYTHON_BIN"
      export FL_DATA_ROOT
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export DEVICE_ID=0
      export MPLCONFIGDIR="$MPL_ROOT/${method}_${dataset_key}_seed${RUN_SEED}"
      export MAX_PARALLEL_JOBS=1
      export SEED="$RUN_SEED"
      export ALGORITHMS="$method"
      export STREAM_LOGS=0
      bash "$dataset_script"
    ) > "$LAST_LOG" 2>&1 &
  fi

  LAST_PID=$!
  echo "[LAUNCH] batch=$batch_number method=$method dataset=$dataset_key seed=$RUN_SEED gpu=$gpu_id pid=$LAST_PID"
  echo "         log=$LAST_LOG"
}

echo "============================================================"
echo "[RQ3 DYNAMIC-CLIENT BASELINE EXPERIMENT]"
echo "objective=Compare FL baselines before and after newcomer clients join"
echo "model=$EXPERIMENT_MODEL"
echo "datasets=CIFAR-10,FashionMNIST"
echo "dataset_paths=Cifar10_dynamic_clients_nc50,FashionMNIST_dynamic_clients_nc50"
echo "client_schedule=rounds 0-50: clients 0-29/classes 0-5"
echo "                round 51+: add clients 30-49/classes 6-9"
echo "metrics=existing/newcomer ID-OOD accuracy and communication cost"
echo "global_rounds=$EXPERIMENT_GLOBAL_ROUNDS"
echo "local_epochs=$EXPERIMENT_LOCAL_EPOCHS"
echo "learning_rate=$EXPERIMENT_LR"
echo "local_batch_size=$EXPERIMENT_BATCH_SIZE"
echo "join_ratio=$EXPERIMENT_JOIN_RATIO"
echo "num_clients=$EXPERIMENT_NUM_CLIENTS"
echo "seed=$RUN_SEED"
echo "methods=${METHODS[*]}"
echo "total_experiments=$((2 * ${#METHODS[@]})) (${#METHODS[@]} methods x 2 datasets)"
echo "parallel_methods=$MAX_PARALLEL_METHODS"
echo "max_concurrent_experiments=$((2 * MAX_PARALLEL_METHODS))"
echo "physical_gpu_pool=${GPU_ID_LIST[*]}"
echo "live_logs=$LIVE_LOGS"
echo "------------------------------------------------------------"
echo "baseline_dir=$BASELINE_DIR"
echo "fl_data_root=$FL_DATA_ROOT"
echo "python=$FEDCD_PYTHON_BIN"
echo "status=$STATUS_TSV"
echo "============================================================"

TOTAL_METHODS="${#METHODS[@]}"
TOTAL_BATCHES=$(( (TOTAL_METHODS + MAX_PARALLEL_METHODS - 1) / MAX_PARALLEL_METHODS ))
FAILED=0

for ((batch_start = 0, batch_number = 1; batch_start < TOTAL_METHODS; batch_start += MAX_PARALLEL_METHODS, batch_number += 1)); do
  batch_end=$((batch_start + MAX_PARALLEL_METHODS))
  if ((batch_end > TOTAL_METHODS)); then
    batch_end=$TOTAL_METHODS
  fi

  echo ""
  echo "[BATCH $batch_number/$TOTAL_BATCHES] methods=${METHODS[*]:batch_start:batch_end-batch_start}"

  PIDS=()
  LABEL_METHODS=()
  LABEL_DATASETS=()
  LABEL_GPUS=()
  LOG_PATHS=()
  START_TIMES=()
  launch_slot=0

  for ((method_idx = batch_start; method_idx < batch_end; method_idx += 1)); do
    method="${METHODS[$method_idx]}"
    for dataset_key in cifar10 fashionmnist; do
      gpu_id="${GPU_ID_LIST[$((launch_slot % ${#GPU_ID_LIST[@]}))]}"
      launch_dataset "$batch_number" "$method" "$dataset_key" "$gpu_id"
      PIDS+=("$LAST_PID")
      LABEL_METHODS+=("$method")
      LABEL_DATASETS+=("$dataset_key")
      LABEL_GPUS+=("$gpu_id")
      LOG_PATHS+=("$LAST_LOG")
      START_TIMES+=("$LAST_START")
      ACTIVE_PIDS+=("$LAST_PID")
      launch_slot=$((launch_slot + 1))

      # The underlying runner uses a second-resolution queue directory that
      # does not contain the dataset or method name. Stagger wrapper starts so
      # concurrent runs do not truncate the same status.csv.
      if ((method_idx < batch_end - 1)) || [[ "$dataset_key" == "cifar10" ]]; then
        sleep "$LAUNCH_STAGGER_SECONDS"
      fi
    done
  done

  batch_failed=0
  for job_idx in "${!PIDS[@]}"; do
    pid="${PIDS[$job_idx]}"
    exit_code=0
    if wait "$pid"; then
      status="ok"
    else
      exit_code=$?
      status="failed"
      batch_failed=1
      FAILED=1
    fi
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$batch_number" \
      "${LABEL_METHODS[$job_idx]}" \
      "${LABEL_DATASETS[$job_idx]}" \
      "$RUN_SEED" \
      "${LABEL_GPUS[$job_idx]}" \
      "$status" \
      "$exit_code" \
      "${START_TIMES[$job_idx]}" \
      "$end_utc" \
      "${LOG_PATHS[$job_idx]}" >> "$STATUS_TSV"

    echo "[${status^^}] method=${LABEL_METHODS[$job_idx]} dataset=${LABEL_DATASETS[$job_idx]} exit=$exit_code"
    if [[ "$status" == "failed" ]]; then
      tail -n 40 "${LOG_PATHS[$job_idx]}" || true
    fi
  done
  ACTIVE_PIDS=()

  if ((batch_failed != 0)); then
    echo "[STOP] Batch $batch_number failed. Remaining methods were not started." >&2
    break
  fi
done

echo ""
echo "[RESULT] status=$STATUS_TSV"
if ((FAILED != 0)); then
  exit 1
fi
echo "[RQ3 COMPLETE] All selected methods and both datasets finished successfully."
