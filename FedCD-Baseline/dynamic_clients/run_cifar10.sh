#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODEL="${MODEL:-VGG8}"
export DATASETS="Cifar10"
export ALGORITHMS="${ALGORITHMS:-FedAvg,FedProx,FedCross,FedBN,FedALA,FedAS,pFedMe,cwFedAvg,FedDST,FedCP,DualFed}"
export DATASET_NAME_OVERRIDE="Cifar10_dynamic_clients_nc50"
export ENABLE_MULTI_RHO_EVAL="false"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
export DYNAMIC_CLIENT_ENABLED="true"
export DYNAMIC_CLIENT_JOIN_ROUND="51"
export DYNAMIC_CLIENT_OLD_CLASSES="0,1,2,3,4,5"
export DYNAMIC_CLIENT_NEW_CLASSES="6,7,8,9"

exec bash "$BASELINE_DIR/run_splitgp_multirho_resnet10.sh"
