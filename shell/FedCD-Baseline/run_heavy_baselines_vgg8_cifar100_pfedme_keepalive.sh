#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$SCRIPT_DIR/batch_runs/pfedme_debug_cifar100_keepalive"
mkdir -p "$LOG_ROOT"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_ROOT/run_${RUN_TS}.log"

echo "[keepalive] CIFAR-100 pFedMe log: $LOG_PATH"
echo "[keepalive] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

bash "$SCRIPT_DIR/run_heavy_baselines_vgg8_cifar100_pfedme.sh" 2>&1 | tee "$LOG_PATH"
status=${PIPESTATUS[0]}

echo
echo "[keepalive] end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[keepalive] exit_code=$status"
echo "[keepalive] tmux session stays open for log inspection."
exec bash
