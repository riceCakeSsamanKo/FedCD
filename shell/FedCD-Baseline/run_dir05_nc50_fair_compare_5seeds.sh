#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
SYSTEM_DIR="$REPO_DIR/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="$(date -u +%H%M%S)"
QUEUE_ROOT="$REPO_DIR/batch_runs/fair_dir05_nc50/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"
STATUS_CSV="$QUEUE_ROOT/status.csv"
echo "group,algorithm,seed,status,exit_code,start_utc,end_utc" > "$STATUS_CSV"

DATASET="Cifar10_dir0.5_nc50"
MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="0"
GLOBAL_ROUNDS="100"
LR="0.005"
LBS="128"
LOCAL_EPOCHS="2"
JOIN_RATIO="1.0"
NUM_CLIENTS="50"
NUM_CLASSES="10"
SEEDS=(0 1 2 3 4)

run_baseline() {
    local algo="$1"
    local seed="$2"
    shift 2
    local goal="fair_${algo}_dir05_nc50_seed${seed}_${DATE_STR}_${TIME_STR}"
    local start_utc end_utc exit_code=0
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[START] baseline algo=${algo} seed=${seed} dataset=${DATASET}"
    (
        cd "$SYSTEM_DIR"
        "$PYTHON_BIN" -u main.py \
            -data "$DATASET" \
            -ncl "$NUM_CLASSES" \
            -m "$MODEL" \
            -algo "$algo" \
            -gr "$GLOBAL_ROUNDS" \
            -lr "$LR" \
            -lbs "$LBS" \
            -ls "$LOCAL_EPOCHS" \
            -nc "$NUM_CLIENTS" \
            -jr "$JOIN_RATIO" \
            -t 1 \
            --seed "$seed" \
            -go "$goal" \
            -dev "$DEVICE" \
            -did "$DEVICE_ID" \
            "$@"
    )
    exit_code=$?
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ $exit_code -eq 0 ]] && status="ok" || status="failed"
    echo "baseline,${algo},${seed},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
}

run_ours() {
    local seed="$1"
    local start_utc end_utc exit_code=0
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[START] ours(t5) seed=${seed} dataset=${DATASET}"
    /home/mulsoap0504/PFLlib-fresh/shell/run_fedccmv12_common_cifar10_nc50.sh Cifar10_dir0.5_nc50 \
      --fedccm_version_tag v12-t5 \
      --fedccmv9_threshold_objective seen_budget \
      --fedccmv9_detector_seen_budget 0.080 \
      --fedccmv9_entropy_search_grid 0.82,0.84,0.86,0.88,0.90 \
      --fedccmv9_maxprob_search_grid 0.08,0.10,0.12,0.15,0.18 \
      --fedccmv9_pm_margin_search_grid 0.12,0.15,0.18,0.22,0.26 \
      --fedccmv10_pm_pool_source cluster_pm \
      --fedccmv9_server_pm_selection weighted_hybrid \
      --fedccmv9_server_pm_prune_by_class True \
      --fedccmv9_server_pm_topk_classes 2 \
      --fedccmv9_server_pm_hybrid_conf_weight 0.65 \
      --fedccmv9_server_pm_hybrid_margin_weight 0.95 \
      --fedccmv9_server_pm_hybrid_entropy_weight 0.40 \
      --fedccmv9_server_pm_hybrid_proto_weight 1.35 \
      -go "fair_FedCCMV12_t5_dir05_nc50_seed${seed}_${DATE_STR}_${TIME_STR}" \
      --seed "$seed"
    exit_code=$?
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ $exit_code -eq 0 ]] && status="ok" || status="failed"
    echo "ours,FedCCMV12_t5,${seed},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
}

for seed in "${SEEDS[@]}"; do
    run_baseline "FedAvg" "$seed"
    run_baseline "FedProx" "$seed" -mu 1.0
    run_baseline "FedBN" "$seed"
    run_baseline "cwFedAvg" "$seed" -cw -wdr -plt -ncw 1 -wd 10
    run_ours "$seed"
done

echo "[INFO] Fair DIR(0.5) NC50 5-seed queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
