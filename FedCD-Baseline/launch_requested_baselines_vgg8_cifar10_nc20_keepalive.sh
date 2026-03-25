#!/bin/bash

set -uo pipefail

SESSION_NAME="${1:-fedcd-baselines}"
WINDOW_NAME="${2:-VGG8-C10-NC20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_requested_baselines_vgg8_cifar10_nc20.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "[ERROR] Run script not found: $RUN_SCRIPT"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME"
tmux send-keys -t "${SESSION_NAME}:0" "cd $SCRIPT_DIR" C-m
tmux send-keys -t "${SESSION_NAME}:0" "bash \"$RUN_SCRIPT\"; EXIT_CODE=\$?; echo; echo \"[EXIT] requested baseline queue finished with code \$EXIT_CODE\"; exec bash" C-m

echo "Started tmux session '$SESSION_NAME' window '$WINDOW_NAME'"
