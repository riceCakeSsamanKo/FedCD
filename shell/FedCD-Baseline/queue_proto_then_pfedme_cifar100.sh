#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_SCRIPT="$SCRIPT_DIR/run_fedproto_remaining_vgg8_cifar100_nc50.sh"
PFEDME_SCRIPT="$SCRIPT_DIR/run_heavy_baselines_vgg8_cifar100_pfedme.sh"

wait_for_gpu_idle() {
    local util_threshold="${GPU_IDLE_UTIL_THRESHOLD:-20}"
    local mem_threshold="${GPU_IDLE_MEM_THRESHOLD:-12000}"
    local sleep_s="${GPU_IDLE_SLEEP_SECONDS:-180}"
    local stable_target="${GPU_IDLE_STABLE_CHECKS:-3}"
    local stable=0

    while true; do
        local line util mem
        line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n 1)"
        util="$(echo "$line" | cut -d',' -f1 | tr -d ' ')"
        mem="$(echo "$line" | cut -d',' -f2 | tr -d ' ')"
        echo "[queue:proto->pfedme-cifar100] GPU util=${util}% mem=${mem}MiB (need util<=${util_threshold}, mem<=${mem_threshold})"
        if [[ "$util" =~ ^[0-9]+$ ]] && [[ "$mem" =~ ^[0-9]+$ ]] && [[ "$util" -le "$util_threshold" ]] && [[ "$mem" -le "$mem_threshold" ]]; then
            stable=$((stable + 1))
            if [[ "$stable" -ge "$stable_target" ]]; then
                break
            fi
        else
            stable=0
        fi
        sleep "$sleep_s"
    done
}

echo "[queue:proto->pfedme-cifar100] waiting for GPU idle before FedProto"
wait_for_gpu_idle

echo "[queue:proto->pfedme-cifar100] starting remaining FedProto"
bash "$PROTO_SCRIPT"
proto_exit=$?
echo "[queue:proto->pfedme-cifar100] FedProto exit=$proto_exit"

if [[ $proto_exit -ne 0 ]]; then
    exit $proto_exit
fi

echo "[queue:proto->pfedme-cifar100] waiting for GPU idle before pFedMe"
wait_for_gpu_idle

echo "[queue:proto->pfedme-cifar100] starting pFedMe"
bash "$PFEDME_SCRIPT"
