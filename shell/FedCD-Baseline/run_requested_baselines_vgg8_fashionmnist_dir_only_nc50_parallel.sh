#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a sessions=(
  "FashionMNIST-dir-nc50-gpu2 worker2 2 FedAS FedAvg FedProx Ditto"
  "FashionMNIST-dir-nc50-gpu3 worker3 3 FedBN FedALA FedCross cwFedAvg"
)

for spec in "${sessions[@]}"; do
  read -r session worker gpu algo1 algo2 algo3 algo4 <<<"$spec"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "bash \"$SCRIPT_DIR/run_requested_baselines_vgg8_fashionmnist_dir_only_nc50_worker.sh\" \"$gpu\" \"$worker\" \"$algo1\" \"$algo2\" \"$algo3\" \"$algo4\" 2>&1 | tee \"$SCRIPT_DIR/${session}.log\""
  echo "[STARTED] $session on GPU $gpu with $algo1 $algo2 $algo3 $algo4"
done
