#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="run_heavy_baselines_vgg8_cifar100_cross_gpfl.sh"
NEXT="$SCRIPT_DIR/run_heavy_baselines_vgg8_cifar100_pfedme.sh"
echo "[queue:pfedme-cifar100] waiting for $TARGET to finish"
while pgrep -af "$TARGET" >/dev/null; do
    sleep 60
done
echo "[queue:pfedme-cifar100] starting pFedMe queue"
exec "$NEXT"
