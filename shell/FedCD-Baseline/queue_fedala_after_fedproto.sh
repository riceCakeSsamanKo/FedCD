#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[queue:FedALA] waiting for FedProto batch to finish..."
while pgrep -f "run_fedproto_vgg8_cifar10_nc20.sh" >/dev/null; do
    sleep 60
done

echo "[queue:FedALA] FedProto finished. Starting FedALA batch."
cd "$SCRIPT_DIR"
bash run_fedala_vgg8_cifar10_nc20.sh
