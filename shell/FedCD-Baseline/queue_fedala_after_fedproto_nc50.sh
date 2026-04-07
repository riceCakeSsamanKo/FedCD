#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[queue:FedALA-NC50] waiting for FedProto NC50 batch to finish..."
while pgrep -f "run_fedproto_remaining_vgg8_cifar10_nc50.sh" >/dev/null; do
    sleep 60
done

echo "[queue:FedALA-NC50] FedProto finished. Starting FedALA NC50 batch."
cd "$SCRIPT_DIR"
bash run_fedala_vgg8_cifar10_nc50.sh
