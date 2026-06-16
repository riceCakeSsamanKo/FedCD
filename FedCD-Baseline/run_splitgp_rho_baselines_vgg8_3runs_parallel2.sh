#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/ds1home/aislab/miniconda3/envs/pfllib/bin/python}"
FL_DATA_ROOT="${FL_DATA_ROOT:-/ds1home/aislab/Min/data/fl_data}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FL_DATA_ROOT

MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="${DEVICE_ID:-0}"
GLOBAL_ROUNDS="100"
LR="0.005"
LBS="128"
LOCAL_EPOCHS="2"
JOIN_RATIO="1.0"
TIMES="1"
NUM_CLIENTS="50"
TARGET_RUNS="${TARGET_RUNS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
WORKER_ID="${WORKER_ID:-${1:-0}}"
NUM_WORKERS="${NUM_WORKERS:-${2:-1}}"

# Completed old runs are treated as run 1. Missing runs use these seeds.
seeds=(1 2 3)

datasets=("Cifar10" "FashionMNIST")
algorithms=("cwFedAvg" "FedALA" "FedAS" "FedAvg" "FedBN" "FedCross" "pFedMe" "FedProx")
rhos=("0.0" "0.2" "0.4" "0.6" "0.8")

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ ]] || ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$NUM_WORKERS" -lt 1 ]]; then
  echo "[ERROR] WORKER_ID and NUM_WORKERS must be non-negative integers, with NUM_WORKERS >= 1." >&2
  exit 1
fi

if [[ "$WORKER_ID" -ge "$NUM_WORKERS" ]]; then
  echo "[ERROR] WORKER_ID=$WORKER_ID must be smaller than NUM_WORKERS=$NUM_WORKERS." >&2
  exit 1
fi

DATE_STR="${RUN_DATE_STR:-$(date -u +%Y%m%d)}"
TIME_STR="${RUN_TIME_STR:-$(date -u +%H%M%S)}"
RUN_TAG="${RUN_TAG:-splitgp_rho_3runs_${DATE_STR}_${TIME_STR}}"
QUEUE_PARENT="$SCRIPT_DIR/batch_runs/splitgp_rho_baselines_vgg8_3runs"
if [[ "$NUM_WORKERS" -gt 1 ]]; then
  QUEUE_ROOT="$QUEUE_PARENT/date_${DATE_STR}/time_${TIME_STR}/worker_${WORKER_ID}"
  SCHEDULER_LOCK="$QUEUE_PARENT/.scheduler.worker_${WORKER_ID}.lock"
else
  QUEUE_ROOT="$QUEUE_PARENT/date_${DATE_STR}/time_${TIME_STR}"
  SCHEDULER_LOCK="$QUEUE_PARENT/.scheduler.lock"
fi
RUN_LOG_DIR="$QUEUE_ROOT/run_logs"
ITEM_ROOT="$QUEUE_ROOT/items"
MPL_ROOT="$QUEUE_ROOT/mpl"
QUEUE_TSV="$QUEUE_ROOT/queue.tsv"
PLAN_CSV="$QUEUE_ROOT/plan.csv"
STATUS_CSV="$QUEUE_ROOT/status.csv"
STATUS_LOCK="$QUEUE_ROOT/status.lock"

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

if [[ "$TARGET_RUNS" -gt "${#seeds[@]}" ]]; then
  echo "[ERROR] TARGET_RUNS=$TARGET_RUNS exceeds configured seeds: ${seeds[*]}" >&2
  exit 1
fi

if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "[ERROR] MAX_PARALLEL must be >= 1" >&2
  exit 1
fi

mkdir -p "$QUEUE_PARENT" "$RUN_LOG_DIR" "$ITEM_ROOT" "$MPL_ROOT"

exec 9>"$SCHEDULER_LOCK"
if ! flock -n 9; then
  echo "[ERROR] Another splitgp rho 3-run scheduler is already running for worker $WORKER_ID/$NUM_WORKERS." >&2
  echo "[ERROR] Lock file: $SCHEDULER_LOCK" >&2
  exit 1
fi

printf '%s\n' 'dataset_base,algorithm,rho,num_clients,complete_before,queued_runs,existing_acc_csvs' > "$PLAN_CSV"
printf '%s\n' 'idx,total_jobs,global_job_idx,worker_id,num_workers,dataset_base,algorithm,rho,num_clients,seed,run_ordinal,target_runs,dataset,status,exit_code,start_utc,end_utc,run_log,complete_before,complete_after,latest_complete_acc_csv' > "$STATUS_CSV"
: > "$QUEUE_TSV"

acc_csv_is_complete() {
  local acc_csv="$1"
  [[ -f "$acc_csv" ]] || return 1
  awk -F',' '
    NR > 1 && $1 ~ /^[0-9]+$/ {
      count += 1
      if ($1 + 0 > max_round) {
        max_round = $1 + 0
      }
    }
    END {
      exit !(count >= 101 && max_round >= 101)
    }
  ' "$acc_csv"
}

log_dataset_name() {
  local dataset_base="$1"
  if [[ "$dataset_base" == "Cifar10" ]]; then
    printf '%s\n' "cifar10"
  else
    printf '%s\n' "$dataset_base"
  fi
}

find_complete_acc_csvs() {
  local dataset_base="$1"
  local algo="$2"
  local rho="$3"
  local log_dataset
  log_dataset="$(log_dataset_name "$dataset_base")"

  local acc_csv
  shopt -s nullglob
  for acc_csv in "$SCRIPT_DIR/logs/$log_dataset/$algo/GM_${MODEL}/splitgp_rho${rho}/NC_${NUM_CLIENTS}"/date_*/time_*/acc.csv; do
    if acc_csv_is_complete "$acc_csv"; then
      printf '%s\n' "$acc_csv"
    fi
  done
  shopt -u nullglob
}

count_complete_acc_csvs() {
  find_complete_acc_csvs "$1" "$2" "$3" | wc -l | awk '{print $1}'
}

join_lines_with_semicolon() {
  local sep=""
  local line
  while IFS= read -r line; do
    printf '%s%s' "$sep" "$line"
    sep=";"
  done
}

append_status() {
  local line="$1"
  {
    flock 8
    printf '%s\n' "$line" >> "$STATUS_CSV"
  } 8>>"$STATUS_LOCK"
}

extra_args_for_algo() {
  local algo="$1"
  case "$algo" in
    FedProx)
      printf '%s\n' "-mu" "1.0"
      ;;
    FedALA)
      printf '%s\n' "-et" "1.0" "-s" "80" "-p" "2"
      ;;
    FedCross)
      printf '%s\n' "-fsb" "0" "-ca" "0.99" "-cmss" "1"
      ;;
    cwFedAvg)
      printf '%s\n' "-cw" "-wdr" "-plt" "-ncw" "1" "-wd" "10"
      ;;
  esac
}

build_queue() {
  local queue_idx=0
  local global_job_idx=0
  local dataset_base algo rho dataset complete_before queued_runs existing_acc_csvs run_ordinal seed

  for dataset_base in "${datasets[@]}"; do
    for algo in "${algorithms[@]}"; do
      for rho in "${rhos[@]}"; do
        dataset="${dataset_base}_splitgp_pat_rho${rho}_nc${NUM_CLIENTS}"
        if [[ ! -d "$FL_DATA_ROOT/$dataset" ]]; then
          echo "[ERROR] Missing dataset: $FL_DATA_ROOT/$dataset" >&2
          exit 1
        fi

        complete_before="$(count_complete_acc_csvs "$dataset_base" "$algo" "$rho")"
        existing_acc_csvs="$(find_complete_acc_csvs "$dataset_base" "$algo" "$rho" | join_lines_with_semicolon)"
        if [[ "$complete_before" -ge "$TARGET_RUNS" ]]; then
          queued_runs=0
        else
          queued_runs=$((TARGET_RUNS - complete_before))
        fi

        printf '%s\n' "${dataset_base},${algo},${rho},${NUM_CLIENTS},${complete_before},${queued_runs},${existing_acc_csvs}" >> "$PLAN_CSV"

        for ((run_ordinal=1; run_ordinal<=TARGET_RUNS; run_ordinal++)); do
          global_job_idx=$((global_job_idx + 1))
          if [[ "$run_ordinal" -le "$complete_before" ]]; then
            continue
          fi
          if [[ $(( (global_job_idx - 1) % NUM_WORKERS )) -ne "$WORKER_ID" ]]; then
            continue
          fi
          seed="${seeds[$((run_ordinal - 1))]}"
          queue_idx=$((queue_idx + 1))
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$queue_idx" "$global_job_idx" "$dataset_base" "$algo" "$rho" "$seed" "$run_ordinal" "$complete_before" "$dataset" >> "$QUEUE_TSV"
        done
      done
    done
  done
}

run_one() {
  local idx="$1"
  local total_jobs="$2"
  local global_job_idx="$3"
  local dataset_base="$4"
  local algo="$5"
  local rho="$6"
  local seed="$7"
  local run_ordinal="$8"
  local complete_before="$9"
  local dataset="${10}"
  local safe_rho="${rho//./p}"
  local start_utc end_utc exit_code status run_log goal item_dir mpl_dir complete_now complete_after latest_acc
  local -a extra_args

  complete_now="$(count_complete_acc_csvs "$dataset_base" "$algo" "$rho")"
  if [[ "$complete_now" -ge "$TARGET_RUNS" || "$run_ordinal" -le "$complete_now" ]]; then
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    end_utc="$start_utc"
    latest_acc="$(find_complete_acc_csvs "$dataset_base" "$algo" "$rho" | tail -n 1)"
    append_status "${idx},${total_jobs},${global_job_idx},${WORKER_ID},${NUM_WORKERS},${dataset_base},${algo},${rho},${NUM_CLIENTS},${seed},${run_ordinal},${TARGET_RUNS},${dataset},skipped_completed,0,${start_utc},${end_utc},,${complete_before},${complete_now},${latest_acc}"
    echo "[SKIP $idx/$total_jobs g${global_job_idx}][worker $WORKER_ID/$NUM_WORKERS] already has ${complete_now}/${TARGET_RUNS}: dataset=$dataset algo=$algo rho=$rho run=$run_ordinal"
    return 0
  fi

  goal="${algo}_splitgp_rho${rho}_nc${NUM_CLIENTS}_${RUN_TAG}_job${global_job_idx}_seed${seed}_run${run_ordinal}"
  run_log="$RUN_LOG_DIR/g${global_job_idx}_q${idx}_${dataset_base}_${algo}_rho${safe_rho}_nc${NUM_CLIENTS}_seed${seed}_run${run_ordinal}.log"
  item_dir="$ITEM_ROOT/job_${global_job_idx}_${dataset_base}_${algo}_rho${safe_rho}_seed${seed}"
  mpl_dir="$MPL_ROOT/job_${global_job_idx}"
  mkdir -p "$item_dir" "$mpl_dir"

  mapfile -t extra_args < <(extra_args_for_algo "$algo")

  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "=========================================================="
  echo "[START $idx/$total_jobs g${global_job_idx}][worker $WORKER_ID/$NUM_WORKERS] dataset=$dataset algo=$algo rho=$rho seed=$seed run=$run_ordinal/$TARGET_RUNS"
  echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO nc=$NUM_CLIENTS"
  echo "[LOG] $run_log"
  echo "=========================================================="

  (
    cd "$SYSTEM_DIR" || exit 1
    MPLCONFIGDIR="$mpl_dir" "$PYTHON_BIN" -u main.py \
      -data "$dataset" \
      -ncl 10 \
      -m "$MODEL" \
      -algo "$algo" \
      -gr "$GLOBAL_ROUNDS" \
      -lr "$LR" \
      -lbs "$LBS" \
      -ls "$LOCAL_EPOCHS" \
      -nc "$NUM_CLIENTS" \
      -jr "$JOIN_RATIO" \
      -t "$TIMES" \
      --seed "$seed" \
      -go "$goal" \
      -dev "$DEVICE" \
      -did "$DEVICE_ID" \
      -sfn "$item_dir" \
      "${extra_args[@]}"
  ) > "$run_log" 2>&1
  exit_code=$?

  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  complete_after="$(count_complete_acc_csvs "$dataset_base" "$algo" "$rho")"
  latest_acc="$(find_complete_acc_csvs "$dataset_base" "$algo" "$rho" | tail -n 1)"

  if [[ "$exit_code" -eq 0 ]]; then
    status="ok"
    echo "[DONE $idx/$total_jobs g${global_job_idx}][worker $WORKER_ID/$NUM_WORKERS] dataset=$dataset algo=$algo rho=$rho seed=$seed"
  else
    status="failed"
    echo "[FAIL $idx/$total_jobs g${global_job_idx}][worker $WORKER_ID/$NUM_WORKERS] dataset=$dataset algo=$algo rho=$rho seed=$seed exit_code=$exit_code"
    echo "[FAIL] Last 40 log lines:"
    tail -n 40 "$run_log" || true
  fi

  append_status "${idx},${total_jobs},${global_job_idx},${WORKER_ID},${NUM_WORKERS},${dataset_base},${algo},${rho},${NUM_CLIENTS},${seed},${run_ordinal},${TARGET_RUNS},${dataset},${status},${exit_code},${start_utc},${end_utc},${run_log},${complete_before},${complete_after},${latest_acc}"
}

running_jobs() {
  jobs -pr | wc -l | awk '{print $1}'
}

build_queue
total_jobs="$(wc -l < "$QUEUE_TSV" | awk '{print $1}')"

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Plan CSV: $PLAN_CSV"
echo "[INFO] Status CSV: $STATUS_CSV"
echo "[INFO] Queue TSV: $QUEUE_TSV"
echo "[INFO] Run logs: $RUN_LOG_DIR"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[INFO] Device: $DEVICE:$DEVICE_ID"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Worker: $WORKER_ID/$NUM_WORKERS"
echo "[INFO] Target completed acc.csv files per setting: $TARGET_RUNS"
echo "[INFO] Max parallel processes: $MAX_PARALLEL"
echo "[INFO] Queued runs: $total_jobs"
echo "[INFO] Existing completed runs are never overwritten; new runs use unique goals, result names, and log directories."
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1, not launching jobs."
  exit 0
fi

if [[ "$total_jobs" -eq 0 ]]; then
  echo "[INFO] Nothing to run. Every setting already has at least $TARGET_RUNS completed acc.csv files."
  exit 0
fi

while IFS=$'\t' read -r idx global_job_idx dataset_base algo rho seed run_ordinal complete_before dataset; do
  while (( "$(running_jobs)" >= MAX_PARALLEL )); do
    wait -n || true
  done
  run_one "$idx" "$total_jobs" "$global_job_idx" "$dataset_base" "$algo" "$rho" "$seed" "$run_ordinal" "$complete_before" "$dataset" &
done < "$QUEUE_TSV"

while (( "$(running_jobs)" > 0 )); do
  wait -n || true
done

failed_count="$(awk -F',' 'NR > 1 && $14 == "failed" {count += 1} END {print count + 0}' "$STATUS_CSV")"
echo "[INFO] SplitGP rho 3-run baseline queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
echo "[INFO] Failed jobs: $failed_count"

if [[ "$failed_count" -gt 0 ]]; then
  exit 1
fi
