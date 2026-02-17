#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -d "$SCRIPT_DIR/../fl_data" ]; then
    FL_DATA_ROOT="$SCRIPT_DIR/../fl_data"
elif [ -d "$SCRIPT_DIR/../../fl_data" ]; then
    FL_DATA_ROOT="$SCRIPT_DIR/../../fl_data"
else
    FL_DATA_ROOT="$SCRIPT_DIR/../fl_data"
fi

DATASET_NAME="${1:-Cifar10_pat_nc20}"
GLOBAL_ROUNDS="${2:-50}"
GPU_DEVICE="${GPU_DEVICE:-cuda}"

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_CMD="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif [ -x "/data/miniconda3/envs/pfllib/bin/python" ]; then
    PYTHON_CMD="/data/miniconda3/envs/pfllib/bin/python"
else
    echo "[AutoTune][Error] No python interpreter found. Set PYTHON_BIN."
    exit 1
fi

if [[ "$DATASET_NAME" =~ _nc([0-9]+)$ ]]; then
    NUM_CLIENTS="${BASH_REMATCH[1]}"
else
    NUM_CLIENTS="20"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
SEARCH_DIR="logs/FedCD/autotune_${DATASET_NAME}_${STAMP}"
mkdir -p "$SEARCH_DIR"
SUMMARY_CSV="$SEARCH_DIR/summary.csv"
echo "case_name,status,score,pm_local_test_acc,gm_only_global_test_acc,global_test_acc,acc_csv_path" > "$SUMMARY_CSV"

COMMON_ARGS=(
    -data "$DATASET_NAME"
    -algo "FedCD"
    --gm_model "VGG8"
    --pm_model "VGG8"
    --fext_model "SmallFExt"
    --fext_dim "512"
    -gr "$GLOBAL_ROUNDS"
    -nc "$NUM_CLIENTS"
    --join_ratio 1.0
    --cluster_threshold 0.1
    --fedcd_enable_clustering True
    --adaptive_threshold True
    --threshold_step 0.01
    --threshold_step_max 0.1
    --threshold_decay 0.9
    --cluster_period 2
    --pm_period 1
    --global_period 2
    --cluster_sample_size 512
    --max_dynamic_clusters 0
    --local_epochs 5
    -dev "$GPU_DEVICE"
    -nw 0
    --pin_memory True
    --prefetch_factor 2
    --amp True
    --tf32 True
    --gpu_batch_mult 1
    --gpu_batch_max 0
    --log_usage True
    --avoid_oom True
    --eval_common_global True
    --global_test_samples 0
    --common_eval_batch_size 256
    --fedcd_fusion_weight 0.6
    --fedcd_gm_logits_weight 0.0
    --fedcd_pm_logits_weight 0.7
    --fedcd_pm_only_weight 1.5
    --fedcd_gm_lr_scale 0.1
    --fedcd_local_pm_only_objective True
    --fedcd_nc_weight 0.02
    --fedcd_nc_target_corr -0.1
    --fedcd_init_pretrain True
    --fedcd_init_epochs 5
    --fedcd_init_lr 0.005
    --fedcd_init_samples 50000
    --fedcd_init_batch_size 256
    --fedcd_init_ce_weight 1.0
    --fedcd_init_kd_weight 1.0
    --fedcd_init_entropy_weight 0.05
    --fedcd_init_diversity_weight 0.05
    --fedcd_entropy_temp_pm 1.0
    --fedcd_entropy_temp_gm 1.0
    --fedcd_entropy_min_pm_weight 0.5
    --fedcd_entropy_max_pm_weight 0.5
    --fedcd_entropy_gate_tau 0.15
    --fedcd_entropy_pm_bias 0.0
    --fedcd_entropy_gm_bias 0.0
    --fedcd_entropy_disagree_gm_boost 0.0
    --fedcd_entropy_use_class_reliability False
    --fedcd_entropy_reliability_scale 0.8
    --fedcd_entropy_hard_switch_margin 0.35
    --fedcd_entropy_use_ood_gate True
    --fedcd_entropy_ood_scale 1.0
    --fedcd_gate_reliability_ema 0.9
    --fedcd_gate_reliability_samples 512
    --fedcd_gate_feature_ema 0.9
    --fedcd_gate_feature_samples 512
    --fedcd_pm_teacher_proxy_dataset "Cifar100"
    --fedcd_pm_teacher_proxy_root ""
    --fedcd_pm_teacher_proxy_split "train"
    --fedcd_pm_teacher_proxy_download False
    --fedcd_pm_teacher_allow_test_fallback False
    --fedcd_search_enable True
    --fedcd_search_min_rounds 8
    --fedcd_search_patience 6
    --fedcd_search_drop_patience 3
    --fedcd_search_drop_delta 0.003
    --fedcd_search_score_gm_weight 0.75
    --fedcd_search_score_pm_weight 0.25
    --fedcd_search_score_eps 0.0001
    --fedcd_search_min_pm_local_acc 0.55
    --fedcd_search_min_gm_global_acc 0.18
)

run_case() {
    local case_name="$1"
    shift
    local before_list
    local after_list
    before_list="$(mktemp)"
    after_list="$(mktemp)"
    find logs/FedCD -type f -name "acc.csv" 2>/dev/null | sort > "$before_list" || true

    echo ""
    echo "============================================================"
    echo "[AutoTune] Running case: $case_name"
    echo "============================================================"
    set +e
    "$PYTHON_CMD" system/main.py "${COMMON_ARGS[@]}" "$@"
    local status=$?
    set -e

    find logs/FedCD -type f -name "acc.csv" 2>/dev/null | sort > "$after_list" || true
    local acc_path
    acc_path="$(comm -13 "$before_list" "$after_list" | tail -n 1)"
    if [ -z "${acc_path}" ]; then
        acc_path="$(find logs/FedCD -type f -name "acc.csv" -print0 | xargs -0 ls -1t 2>/dev/null | head -n 1)"
    fi

    local metrics
    metrics="$("$PYTHON_CMD" - "$acc_path" <<'PY'
import csv, sys
acc_path = sys.argv[1]
if not acc_path:
    print("0.0,0.0,0.0")
    raise SystemExit(0)
with open(acc_path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("0.0,0.0,0.0")
    raise SystemExit(0)
last = rows[-1]
pm_local = float(last.get("pm_local_test_acc", "0") or 0.0)
gm_global = float(last.get("gm_only_global_test_acc", "0") or 0.0)
global_acc = float(last.get("global_test_acc", "0") or 0.0)
print(f"{pm_local},{gm_global},{global_acc}")
PY
)"
    local pm_local gm_global global_acc score
    pm_local="$(echo "$metrics" | cut -d',' -f1)"
    gm_global="$(echo "$metrics" | cut -d',' -f2)"
    global_acc="$(echo "$metrics" | cut -d',' -f3)"
    score="$("$PYTHON_CMD" - "$pm_local" "$gm_global" <<'PY'
import sys
pm = float(sys.argv[1]); gm = float(sys.argv[2])
print(f"{0.25*pm + 0.75*gm:.6f}")
PY
)"
    local status_text="ok"
    if [ "$status" -ne 0 ]; then
        status_text="failed($status)"
    fi

    echo "[AutoTune] $case_name => score=$score, pm_local=$pm_local, gm_global=$gm_global, global=$global_acc"
    echo "$case_name,$status_text,$score,$pm_local,$gm_global,$global_acc,$acc_path" >> "$SUMMARY_CSV"
    rm -f "$before_list" "$after_list"
}

# 1) Prototype teacher (strong KL)
run_case "proto_strong_kl" \
    --fedcd_gm_update_mode server_proto_teacher \
    --fedcd_proto_teacher_lr 0.005 \
    --fedcd_proto_teacher_steps 220 \
    --fedcd_proto_teacher_batch_size 256 \
    --fedcd_proto_teacher_temp 2.0 \
    --fedcd_proto_teacher_ce_weight 0.5 \
    --fedcd_proto_teacher_kl_weight 1.2 \
    --fedcd_proto_teacher_noise_scale 0.8 \
    --fedcd_proto_teacher_min_count 1 \
    --fedcd_proto_teacher_client_samples 0 \
    --fedcd_proto_teacher_confidence_weight True \
    --fedcd_proto_teacher_confidence_min 0.05 \
    --fedcd_proto_teacher_confidence_power 1.5

# 2) Prototype teacher (more CE anchor)
run_case "proto_balanced_ce" \
    --fedcd_gm_update_mode server_proto_teacher \
    --fedcd_proto_teacher_lr 0.004 \
    --fedcd_proto_teacher_steps 260 \
    --fedcd_proto_teacher_batch_size 256 \
    --fedcd_proto_teacher_temp 2.0 \
    --fedcd_proto_teacher_ce_weight 1.0 \
    --fedcd_proto_teacher_kl_weight 0.8 \
    --fedcd_proto_teacher_noise_scale 0.7 \
    --fedcd_proto_teacher_min_count 1 \
    --fedcd_proto_teacher_client_samples 0 \
    --fedcd_proto_teacher_confidence_weight True \
    --fedcd_proto_teacher_confidence_min 0.05 \
    --fedcd_proto_teacher_confidence_power 1.0

# 3) PM-teacher distillation (top-k selective)
run_case "pm_teacher_topk3" \
    --fedcd_gm_update_mode server_pm_teacher \
    --fedcd_pm_teacher_lr 0.01 \
    --fedcd_pm_teacher_temp 2.2 \
    --fedcd_pm_teacher_kl_weight 1.2 \
    --fedcd_pm_teacher_ce_weight 0.0 \
    --fedcd_pm_teacher_epochs 5 \
    --fedcd_pm_teacher_samples 50000 \
    --fedcd_pm_teacher_batch_size 256 \
    --fedcd_pm_teacher_confidence_weight True \
    --fedcd_pm_teacher_confidence_min 0.10 \
    --fedcd_pm_teacher_confidence_power 2.5 \
    --fedcd_pm_teacher_ensemble_confidence True \
    --fedcd_pm_teacher_topk 3 \
    --fedcd_pm_teacher_abstain_threshold 0.35 \
    --fedcd_pm_teacher_rel_weight 0.2 \
    --fedcd_pm_teacher_rel_batch 64

# 4) PM-teacher distillation (wider teacher pool)
run_case "pm_teacher_topk5" \
    --fedcd_gm_update_mode server_pm_teacher \
    --fedcd_pm_teacher_lr 0.008 \
    --fedcd_pm_teacher_temp 2.0 \
    --fedcd_pm_teacher_kl_weight 1.0 \
    --fedcd_pm_teacher_ce_weight 0.0 \
    --fedcd_pm_teacher_epochs 6 \
    --fedcd_pm_teacher_samples 50000 \
    --fedcd_pm_teacher_batch_size 256 \
    --fedcd_pm_teacher_confidence_weight True \
    --fedcd_pm_teacher_confidence_min 0.10 \
    --fedcd_pm_teacher_confidence_power 2.0 \
    --fedcd_pm_teacher_ensemble_confidence True \
    --fedcd_pm_teacher_topk 5 \
    --fedcd_pm_teacher_abstain_threshold 0.30 \
    --fedcd_pm_teacher_rel_weight 0.1 \
    --fedcd_pm_teacher_rel_batch 64

echo ""
echo "============================================================"
echo "[AutoTune] Finished. Summary: $SUMMARY_CSV"
echo "[AutoTune] Top candidates:"
"$PYTHON_CMD" - "$SUMMARY_CSV" <<'PY'
import csv, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
rows = [r for r in rows if r.get("score")]
rows.sort(key=lambda r: float(r["score"]), reverse=True)
for r in rows[:5]:
    print(f"{r['case_name']}: score={r['score']}, pm_local={r['pm_local_test_acc']}, gm_global={r['gm_only_global_test_acc']}, global={r['global_test_acc']}")
PY
echo "============================================================"
