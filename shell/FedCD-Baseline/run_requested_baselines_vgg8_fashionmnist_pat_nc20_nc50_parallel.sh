#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a sessions=(
  "FashionMNIST-pat-gpu0 worker0 0 FedAS FedAvg"
  "FashionMNIST-pat-gpu1 worker1 1 FedProx Ditto"
  "FashionMNIST-pat-gpu2 worker2 2 FedBN FedALA"
  "FashionMNIST-pat-gpu3 worker3 3 FedCross cwFedAvg"
)

for spec in "${sessions[@]}"; do
  read -r session worker gpu algo1 algo2 <<<"$spec"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "bash \"$SCRIPT_DIR/run_requested_baselines_vgg8_fashionmnist_pat_nc20_nc50_worker.sh\" \"$gpu\" \"$worker\" \"$algo1\" \"$algo2\" 2>&1 | tee \"$SCRIPT_DIR/${session}.log\""
  echo "[STARTED] $session on GPU $gpu with $algo1 $algo2"
done
