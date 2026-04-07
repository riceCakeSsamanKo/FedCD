#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="$(date -u +%H%M%S)"
QUEUE_ROOT="$REPO_DIR/batch_runs/once_cifar100_dir05_nc50_ours/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"
STATUS_CSV="$QUEUE_ROOT/status.csv"
echo "group,algorithm,seed,status,exit_code,start_utc,end_utc" > "$STATUS_CSV"
SEED="0"
start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[START] ours(t5) seed=${SEED} dataset=Cifar100_dir0.5_nc50"
/home/mulsoap0504/PFLlib-fresh/shell/run_fedccmv12_common_cifar100_nc50.sh Cifar100_dir0.5_nc50 \
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
  -go "once_FedCCMV12_t5_cifar100_dir05_nc50_seed${SEED}_${DATE_STR}_${TIME_STR}" \
  --seed "$SEED"
exit_code=$?
end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[[ $exit_code -eq 0 ]] && status="ok" || status="failed"
echo "ours,FedCCMV12_t5,${SEED},${status},${exit_code},${start_utc},${end_utc}" >> "$STATUS_CSV"
echo "[INFO] CIFAR-100 DIR(0.5) NC50 ours-once queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
