#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$SCRIPT_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

WORKER_ID="${WORKER_ID:-${1:-0}}"
NUM_WORKERS="${NUM_WORKERS:-${2:-4}}"
START_IDX="${START_IDX:-21}"
RUN_TAG="${RUN_TAG:-extra2_20260424_162110}"
SKIP_IDXS=",${SKIP_IDXS:-},"
STREAM_RUN_LOGS="${STREAM_RUN_LOGS:-0}"

if [[ ! "$WORKER_ID" =~ ^[0-9]+$ ]] || [[ ! "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$NUM_WORKERS" -le 0 ]]; then
  echo "[ERROR] Usage: WORKER_ID=<0..N-1> NUM_WORKERS=<N> $0" >&2
  exit 1
fi

if [[ "$WORKER_ID" -ge "$NUM_WORKERS" ]]; then
  echo "[ERROR] WORKER_ID must be smaller than NUM_WORKERS" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$SYSTEM_DIR" ]]; then
  echo "[ERROR] System directory not found: $SYSTEM_DIR" >&2
  exit 1
fi

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="${RESUME_TIME_STR:-$(date -u +%H%M%S)}"
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/requested_baselines_vgg8_all_datasets_extra2_resume_parallel4_from21/date_${DATE_STR}/time_${TIME_STR}/worker_${WORKER_ID}"
RUN_LOG_DIR="$QUEUE_ROOT/run_logs"
mkdir -p "$RUN_LOG_DIR"

STATUS_CSV="$QUEUE_ROOT/status.csv"
printf '%s\n' 'idx,total,dataset_base,algorithm,scenario,num_clients,rep,seed,dataset,status,exit_code,start_utc,end_utc,run_log' > "$STATUS_CSV"

MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="${DEVICE_ID:-0}"
GLOBAL_ROUNDS="100"
LR="0.005"
LBS="128"
LOCAL_EPOCHS="2"
JOIN_RATIO="1.0"
TIMES="1"

datasets=("Cifar10" "Cifar100" "FashionMNIST")
algorithms=("FedAS" "FedAvg" "FedProx" "Ditto" "FedBN" "FedALA" "FedCross" "cwFedAvg")
scenarios=("pat" "dir0.1" "dir0.5" "dir1.0")
client_counts=("20" "50")
seeds=("1" "2")

total=$(( ${#datasets[@]} * ${#algorithms[@]} * ${#scenarios[@]} * ${#client_counts[@]} * ${#seeds[@]} ))
idx=0

cd "$SYSTEM_DIR" || exit 1

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Status CSV: $STATUS_CSV"
echo "[INFO] Run logs: $RUN_LOG_DIR"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Device: $DEVICE:$DEVICE_ID"
echo "[INFO] Worker: $WORKER_ID/$NUM_WORKERS"
          echo "[INFO] Resume start idx: $START_IDX"
if [[ "$SKIP_IDXS" != ",," ]]; then
  echo "[INFO] Skip idxs: ${SKIP_IDXS#,}"
fi
if [[ "$STREAM_RUN_LOGS" == "1" ]]; then
  echo "[INFO] Streaming each run log to worker stdout"
fi
echo "[INFO] Goal RUN_TAG: $RUN_TAG"
echo "[INFO] Original total runs: $total"
echo

for dataset_base in "${datasets[@]}"; do
  num_classes="10"
  if [[ "$dataset_base" == "Cifar100" ]]; then
    num_classes="100"
  fi

  for algo in "${algorithms[@]}"; do
    for scenario in "${scenarios[@]}"; do
      for nc in "${client_counts[@]}"; do
        dataset="${dataset_base}_${scenario}_nc${nc}"

        for seed in "${seeds[@]}"; do
          idx=$((idx + 1))
          if [[ "$idx" -lt "$START_IDX" ]]; then
            continue
          fi
          if [[ "$SKIP_IDXS" == *",$idx,"* ]]; then
            echo "[SKIP $idx/$total][worker $WORKER_ID/$NUM_WORKERS] already completed"
            continue
          fi
          if [[ $(( (idx - START_IDX) % NUM_WORKERS )) -ne "$WORKER_ID" ]]; then
            continue
          fi

          rep="$seed"
          goal="${algo}_${scenario}_nc${nc}_${RUN_TAG}_seed${seed}"
          safe_scenario="${scenario//./}"
          run_log="$RUN_LOG_DIR/${idx}_${dataset_base}_${algo}_${safe_scenario}_nc${nc}_seed${seed}.log"
          start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          exit_code=0
          status="ok"
          extra_args=()

          case "$algo" in
            FedProx)
              extra_args+=(-mu 1.0)
              ;;
            Ditto)
              extra_args+=(-mu 1.0 -pls 1)
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
          echo "[START $idx/$total][worker $WORKER_ID/$NUM_WORKERS] dataset=$dataset algo=$algo seed=$seed"
          echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO nc=$nc ncl=$num_classes"
          echo "[LOG] $run_log"
          echo "=========================================================="

          run_cmd=(
            "$PYTHON_BIN" -u main.py
            -data "$dataset"
            -ncl "$num_classes"
            -m "$MODEL"
            -algo "$algo"
            -gr "$GLOBAL_ROUNDS"
            -lr "$LR"
            -lbs "$LBS"
            -ls "$LOCAL_EPOCHS"
            -nc "$nc"
            -jr "$JOIN_RATIO"
            -t "$TIMES"
            --seed "$seed"
            -go "$goal"
            -dev "$DEVICE"
            -did "$DEVICE_ID"
            "${extra_args[@]}"
          )

          if [[ "$STREAM_RUN_LOGS" == "1" ]]; then
            "${run_cmd[@]}" 2>&1 | tee "$run_log"
            exit_code=${PIPESTATUS[0]}
          else
            "${run_cmd[@]}" > "$run_log" 2>&1
            exit_code=$?
          fi

          end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          if [[ $exit_code -eq 0 ]]; then
            status="ok"
            echo "[DONE $idx/$total][worker $WORKER_ID/$NUM_WORKERS] dataset=$dataset algo=$algo seed=$seed"
          else
            status="failed"
            echo "[FAIL $idx/$total][worker $WORKER_ID/$NUM_WORKERS] dataset=$dataset algo=$algo seed=$seed exit_code=$exit_code"
            echo "[FAIL] Last 40 log lines:"
            tail -n 40 "$run_log" || true
          fi

          printf '%s\n' "${idx},${total},${dataset_base},${algo},${scenario},${nc},${rep},${seed},${dataset},${status},${exit_code},${start_utc},${end_utc},${run_log}" >> "$STATUS_CSV"
          echo
        done
      done
    done
  done
done

echo "[INFO] Worker queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
