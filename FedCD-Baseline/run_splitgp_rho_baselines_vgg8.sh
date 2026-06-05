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

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="$(date -u +%H%M%S)"
RUN_TAG="splitgp_rho_${DATE_STR}_${TIME_STR}"
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/splitgp_rho_baselines_vgg8/date_${DATE_STR}/time_${TIME_STR}"
RUN_LOG_DIR="$QUEUE_ROOT/run_logs"
mkdir -p "$RUN_LOG_DIR"

STATUS_CSV="$QUEUE_ROOT/status.csv"
printf '%s\n' 'idx,total,dataset_base,algorithm,rho,num_clients,seed,dataset,status,exit_code,start_utc,end_utc,run_log,existing_acc_csv' > "$STATUS_CSV"

MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="${DEVICE_ID:-0}"
GLOBAL_ROUNDS="100"
LR="0.005"
LBS="128"
LOCAL_EPOCHS="2"
JOIN_RATIO="1.0"
TIMES="1"
SEED="${SEED:-1}"
NUM_CLIENTS="50"

# Deterministic order: 2 datasets * 8 algorithms * 5 rho values = 80 runs.
datasets=("Cifar10" "FashionMNIST")
algorithms=("cwFedAvg" "FedALA" "FedAS" "FedAvg" "FedBN" "FedCross" "pFedMe" "FedProx")
rhos=("0.0" "0.2" "0.4" "0.6" "0.8")

total=$(( ${#datasets[@]} * ${#algorithms[@]} * ${#rhos[@]} ))
idx=0

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

find_complete_acc_csv() {
  local dataset_base="$1"
  local algo="$2"
  local rho="$3"
  local log_dataset="$dataset_base"
  if [[ "$log_dataset" == "Cifar10" ]]; then
    log_dataset="cifar10"
  fi

  local acc_csv
  shopt -s nullglob
  for acc_csv in "$SCRIPT_DIR/logs/$log_dataset/$algo/GM_${MODEL}/splitgp_rho${rho}/NC_${NUM_CLIENTS}"/date_*/time_*/acc.csv; do
    if acc_csv_is_complete "$acc_csv"; then
      printf '%s\n' "$acc_csv"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Status CSV: $STATUS_CSV"
echo "[INFO] Run logs: $RUN_LOG_DIR"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[INFO] Device: $DEVICE:$DEVICE_ID"
echo "[INFO] Total runs: $total"
echo "[INFO] Resume policy: skip runs with an existing acc.csv completed through round 101."
echo

for dataset_base in "${datasets[@]}"; do
  num_classes="10"

  for algo in "${algorithms[@]}"; do
    for rho in "${rhos[@]}"; do
      dataset="${dataset_base}_splitgp_pat_rho${rho}_nc${NUM_CLIENTS}"
      if [[ ! -d "$FL_DATA_ROOT/$dataset" ]]; then
        echo "[ERROR] Missing dataset: $FL_DATA_ROOT/$dataset" >&2
        exit 1
      fi

      idx=$((idx + 1))
      existing_acc_csv=""
      if existing_acc_csv="$(find_complete_acc_csv "$dataset_base" "$algo" "$rho")"; then
        start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        end_utc="$start_utc"
        echo "[SKIP $idx/$total] complete through round 101: dataset=$dataset algo=$algo seed=$SEED"
        echo "[SKIP] existing acc.csv: $existing_acc_csv"
        printf '%s\n' "${idx},${total},${dataset_base},${algo},${rho},${NUM_CLIENTS},${SEED},${dataset},skipped_completed,0,${start_utc},${end_utc},,${existing_acc_csv}" >> "$STATUS_CSV"
        echo
        continue
      fi

      goal="${algo}_splitgp_rho${rho}_nc${NUM_CLIENTS}_${RUN_TAG}_seed${SEED}"
      safe_rho="${rho//./p}"
      run_log="$RUN_LOG_DIR/${idx}_${dataset_base}_${algo}_rho${safe_rho}_nc${NUM_CLIENTS}_seed${SEED}.log"
      start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      exit_code=0
      status="ok"
      extra_args=()

      case "$algo" in
        FedProx)
          extra_args+=(-mu 1.0)
          ;;
        FedALA)
          extra_args+=(-et 1.0 -s 80 -p 2)
          ;;
        FedCross)
          extra_args+=(-fsb 0 -ca 0.99 -cmss 1)
          ;;
        cwFedAvg)
          extra_args+=(-cw -wdr -plt -ncw 1 -wd 10)
          ;;
      esac

      echo "=========================================================="
      echo "[START $idx/$total] dataset=$dataset algo=$algo seed=$SEED"
      echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO nc=$NUM_CLIENTS ncl=$num_classes"
      echo "[LOG] $run_log"
      echo "=========================================================="

      "$PYTHON_BIN" -u main.py \
        -data "$dataset" \
        -ncl "$num_classes" \
        -m "$MODEL" \
        -algo "$algo" \
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
        "${extra_args[@]}" > "$run_log" 2>&1
      exit_code=$?

      end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      if [[ $exit_code -eq 0 ]]; then
        status="ok"
        echo "[DONE $idx/$total] dataset=$dataset algo=$algo seed=$SEED"
      else
        status="failed"
        echo "[FAIL $idx/$total] dataset=$dataset algo=$algo seed=$SEED exit_code=$exit_code"
        echo "[FAIL] Last 40 log lines:"
        tail -n 40 "$run_log" || true
      fi

      printf '%s\n' "${idx},${total},${dataset_base},${algo},${rho},${NUM_CLIENTS},${SEED},${dataset},${status},${exit_code},${start_utc},${end_utc},${run_log}," >> "$STATUS_CSV"
      echo
    done
  done
done

echo "[INFO] SplitGP rho baseline queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
