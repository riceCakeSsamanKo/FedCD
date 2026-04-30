#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-extra2_20260424_162110}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_IDX="${START_IDX:-215}"
STREAM_RUN_LOGS="${STREAM_RUN_LOGS:-1}"
FEDCD_PYTHON="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE_ID="${DEVICE_ID:-0}"

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="${RESUME_TIME_STR:-$(date -u +%H%M%S)}"
WRAPPER_ROOT="$SCRIPT_DIR/batch_runs/requested_baselines_vgg8_all_datasets_extra2_remaining/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$WRAPPER_ROOT"

completed_idxs="$(
  python3 - <<'PY'
from pathlib import Path

root = Path("/home/mulsoap0504/FedCD/FedCD-Baseline")
run_tag = "extra2_20260424_162110"
datasets = ["Cifar10", "Cifar100", "FashionMNIST"]
algorithms = ["FedAS", "FedAvg", "FedProx", "Ditto", "FedBN", "FedALA", "FedCross", "cwFedAvg"]
scenarios = ["pat", "dir0.1", "dir0.5", "dir1.0"]
client_counts = ["20", "50"]
seeds = ["1", "2"]

completed = []
idx = 0
for dataset_base in datasets:
    for algo in algorithms:
        for scenario in scenarios:
            for nc in client_counts:
                for seed in seeds:
                    idx += 1
                    dataset = f"{dataset_base}_{scenario}_nc{nc}"
                    result = root / "results" / f"{dataset}_{algo}_{algo}_{scenario}_nc{nc}_{run_tag}_seed{seed}_0.h5"
                    if result.exists():
                        completed.append(str(idx))

print(",".join(completed))
PY
)"

export RUN_TAG
export START_IDX
export NUM_WORKERS
export SKIP_IDXS="$completed_idxs"
export STREAM_RUN_LOGS
export RESUME_TIME_STR="$TIME_STR"
export FEDCD_PYTHON
export CUDA_VISIBLE_DEVICES
export DEVICE_ID

python3 - <<'PY' > "$WRAPPER_ROOT/missing_before.csv"
from pathlib import Path

root = Path("/home/mulsoap0504/FedCD/FedCD-Baseline")
run_tag = "extra2_20260424_162110"
datasets = ["Cifar10", "Cifar100", "FashionMNIST"]
algorithms = ["FedAS", "FedAvg", "FedProx", "Ditto", "FedBN", "FedALA", "FedCross", "cwFedAvg"]
scenarios = ["pat", "dir0.1", "dir0.5", "dir1.0"]
client_counts = ["20", "50"]
seeds = ["1", "2"]

print("idx,dataset_base,algorithm,scenario,num_clients,seed,result")
idx = 0
for dataset_base in datasets:
    for algo in algorithms:
        for scenario in scenarios:
            for nc in client_counts:
                for seed in seeds:
                    idx += 1
                    dataset = f"{dataset_base}_{scenario}_nc{nc}"
                    result = root / "results" / f"{dataset}_{algo}_{algo}_{scenario}_nc{nc}_{run_tag}_seed{seed}_0.h5"
                    if not result.exists():
                        print(f"{idx},{dataset_base},{algo},{scenario},{nc},{seed},{result}")
PY

missing_count=$(( $(wc -l < "$WRAPPER_ROOT/missing_before.csv") - 1 ))
echo "$completed_idxs" > "$WRAPPER_ROOT/completed_idxs.txt"

echo "[WRAPPER] root=$WRAPPER_ROOT"
echo "[WRAPPER] run_tag=$RUN_TAG start_idx=$START_IDX workers=$NUM_WORKERS missing_before=$missing_count"
echo "[WRAPPER] completed idx list: $WRAPPER_ROOT/completed_idxs.txt"
echo "[WRAPPER] missing list: $WRAPPER_ROOT/missing_before.csv"

pids=()
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
  (
    echo "[WRAPPER] worker=$worker_id start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "$SCRIPT_DIR/run_requested_baselines_vgg8_all_datasets_extra2_resume_parallel4_from21.sh" "$worker_id" "$NUM_WORKERS"
    echo "[WRAPPER] worker=$worker_id end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ) > "$WRAPPER_ROOT/worker_${worker_id}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid" > "$WRAPPER_ROOT/worker_${worker_id}.pid"
  echo "[WRAPPER] launched worker=$worker_id pid=$pid log=$WRAPPER_ROOT/worker_${worker_id}.log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "[WRAPPER] all workers finished status=$status $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"
