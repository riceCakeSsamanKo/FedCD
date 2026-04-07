#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/batch_runs/launch_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/lgfedavg_cifar10_and_cifar100_all_${STAMP}.log"

{
  echo "[LAUNCH] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[STEP] LG-FedAvg on CIFAR-10"
  bash "$SCRIPT_DIR/run_requested_lgfedavg_vgg8_cifar10_all.sh"
  echo "[STEP] All baselines on CIFAR-100"
  bash "$SCRIPT_DIR/run_requested_baselines_cifar100_all.sh"
  echo "[DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$LOG_FILE"

echo "[INFO] launch log: $LOG_FILE"
exec bash
