#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIME_STR="${RESUME_TIME_STR:-$(date -u +%H%M%S)}"
LOG_DIR="${RESUME_WRAPPER_LOG_DIR:-/tmp}"
mkdir -p "$LOG_DIR"

echo "[INFO] Starting fixed resume workers"
echo "[INFO] Repo: $SCRIPT_DIR"
echo "[INFO] Time tag: $TIME_STR"
echo "[INFO] Wrapper logs: $LOG_DIR/fedcd_extra2_resume_fixed_w{0..3}.log"
echo "[INFO] Streaming worker and run logs to this terminal."
echo

pids=()

run_worker() {
  local worker_id="$1"
  local log_file="$LOG_DIR/fedcd_extra2_resume_fixed_w${worker_id}.log"

  (
    cd "$SCRIPT_DIR"
    echo "[WRAPPER] worker=$worker_id log=$log_file"
    WORKER_ID="$worker_id" \
    NUM_WORKERS=4 \
    START_IDX=131 \
    SKIP_IDXS=133,134 \
    RESUME_TIME_STR="$TIME_STR" \
    STREAM_RUN_LOGS=1 \
    bash ./run_requested_baselines_vgg8_all_datasets_extra2_resume_parallel4_from21.sh
  ) 2>&1 | sed -u "s/^/[w${worker_id}] /" | tee "$log_file"
}

for worker_id in 0 1 2 3; do
  run_worker "$worker_id" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
