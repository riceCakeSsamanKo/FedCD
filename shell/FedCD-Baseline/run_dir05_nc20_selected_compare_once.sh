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
QUEUE_ROOT="$REPO_DIR/batch_runs/once_dir05_nc20_selected/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"
STATUS_CSV="$QUEUE_ROOT/status.csv"
echo "group,algorithm,seed,status,exit_code,start_utc,end_utc" > "$STATUS_CSV"
DATASET="Cifar10_dir0.5_nc20"
MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="0"
GLOBAL_ROUNDS="100"
LR="0.005"
LBS="128"
LOCAL_EPOCHS="2"
JOIN_RATIO="1.0"
NUM_CLIENTS="20"
SEED="0"
run_baseline() {
    local algo="$1"
    shift
    local goal="once_${algo}_dir05_nc20_seed${SEED}_${DATE_STR}_${TIME_STR}"
    local start_utc end_utc exit_code=0 status
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[START] baseline algo=${algo} seed=${SEED} dataset=${DATASET}"
    (
        cd "$SYSTEM_DIR"
        "$PYTHON_BIN" -u main.py \
            -data "$DATASET" \
            -m "$MODEL" \
            -algo "$algo" \
            -gr "$GLOBAL_ROUNDS" \
            -lr "$LR" \
            -lbs "$LBS" \
            -ls "$LOCAL_EPOCHS" \
            -nc "$NUM_CLIENTS" \
            -jr "$JOIN_RATIO" \
            -go "$goal" \
            -dev "$DEVICE" \
            -did "$DEVICE_ID" \
            -t 1 \
            --seed "$SEED" \
            "$@"
    )
    exit_code=$?
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ $exit_code -eq 0 ]] && status="ok" || status="failed"
    echo "baseline,${algo},${SEED},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
}
run_ours() {
    local start_utc end_utc exit_code=0 status
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[START] ours(t5) seed=${SEED} dataset=${DATASET}"
    /home/mulsoap0504/PFLlib-fresh/shell/run_fedccmv12_common_cifar10_nc20.sh Cifar10_dir0.5_nc20 \
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
      -go "once_FedCCMV12_t5_dir05_nc20_seed${SEED}_${DATE_STR}_${TIME_STR}" \
      --seed "$SEED"
    exit_code=$?
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ $exit_code -eq 0 ]] && status="ok" || status="failed"
    echo "ours,FedCCMV12_t5,${SEED},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
}
run_baseline FedProx -mu 1.0
run_baseline FedBN
run_baseline cwFedAvg -cw -wdr -plt -ncw 1 -wd 10
run_ours
