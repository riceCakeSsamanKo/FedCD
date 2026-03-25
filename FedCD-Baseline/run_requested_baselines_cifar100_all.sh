#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_requested_baselines_vgg8_cifar100_nc20.sh"
bash "$SCRIPT_DIR/run_requested_baselines_vgg8_cifar100_nc50.sh"
