#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

SCENARIO_FILTER="dir1.0" MAX_CONCURRENT=1 bash "$SCRIPT_DIR/run_grid_best_setting.sh" "$@"
