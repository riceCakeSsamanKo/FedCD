#!/bin/bash

set -uo pipefail

SESSION_NAME="${1:-fedcd-baselines-mnist}"
WINDOW_NAME="${2:-VGG8-MNIST-ALL}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_requested_baselines_vgg8_mnist_all.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "[ERROR] Run script not found: $RUN_SCRIPT"
    exit 1
fi

RUN_CMD="cd \"$SCRIPT_DIR\" && bash \"$RUN_SCRIPT\"; EXIT_CODE=\$?; echo; echo \"[EXIT] requested MNIST baseline queues finished with code \$EXIT_CODE\"; exec bash"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-window -t "$SESSION_NAME" -n "$WINDOW_NAME" "bash -lc '$RUN_CMD'"
else
    tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" "bash -lc '$RUN_CMD'"
fi

echo "Started tmux session '$SESSION_NAME' window '$WINDOW_NAME'"
