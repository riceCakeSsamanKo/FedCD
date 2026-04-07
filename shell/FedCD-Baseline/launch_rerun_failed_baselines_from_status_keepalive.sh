#!/bin/bash

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <status_csv_path> [session_name] [window_name]"
    exit 1
fi

STATUS_INPUT="$1"
SESSION_NAME="${2:-fedcd-baselines}"
WINDOW_NAME="${3:-Retry-Failed}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/rerun_failed_baselines_from_status.sh"

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "[ERROR] Run script not found: $RUN_SCRIPT"
    exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-window -t "$SESSION_NAME" -n "$WINDOW_NAME"
    TARGET="${SESSION_NAME}:$WINDOW_NAME"
else
    tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME"
    TARGET="${SESSION_NAME}:0"
fi

tmux send-keys -t "$TARGET" "cd $SCRIPT_DIR" C-m
tmux send-keys -t "$TARGET" "bash \"$RUN_SCRIPT\" \"$STATUS_INPUT\"; EXIT_CODE=\$?; echo; echo \"[EXIT] failed baseline rerun finished with code \$EXIT_CODE\"; exec bash" C-m

echo "Started tmux target '$TARGET'"
