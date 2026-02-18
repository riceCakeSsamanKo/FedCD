#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Python executable detection
if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_CMD="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif [ -x "/data/miniconda3/envs/pfllib/bin/python" ]; then
    PYTHON_CMD="/data/miniconda3/envs/pfllib/bin/python"
else
    echo "[Error] Python interpreter not found. Set PYTHON_BIN."
    exit 1
fi

# ------------------------------------------------------------------------------
# Base setting: FedCD-Core/logs/FedCD/.../time_194429
# Minimal arguments only + PM/GM-related sweep knobs.
# ------------------------------------------------------------------------------
GPU_ID="${GPU_ID:-0}"                 # single physical GPU
MAX_CONCURRENT="${MAX_CONCURRENT:-2}" # same GPU concurrent jobs

GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
TOTAL_DATA=50000

# Scenarios (same coverage as run.sh)
NCS=(20 50)
SCENARIOS=("pat" "dir0.1" "dir0.5" "dir1.0")

# Sweep knobs (PM/GM-centric)
LOCAL_LRS=(0.004 0.005 0.006)
PM_TEACHER_LRS=(0.006 0.008)
PM_TEACHER_EPOCHS=(4 6)

# Fixed best-known values from time_194429
PM_TEACHER_TEMP=2.0
PM_TEACHER_KL_WEIGHT=1.0
PM_TEACHER_CE_WEIGHT=0.0
PM_TEACHER_SAMPLES=50000
PM_TEACHER_BATCH_SIZE=256
PM_TEACHER_TOPK=5
PM_TEACHER_ABSTAIN=0.3
PM_TEACHER_CONF_MIN=0.1
PM_TEACHER_CONF_POWER=2.0
PM_TEACHER_REL_WEIGHT=0.1
PM_TEACHER_REL_BATCH=64

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="logs/FedCD/grid_194429_${STAMP}"
mkdir -p "$RUN_ROOT"
MANIFEST="$RUN_ROOT/manifest.csv"

echo "exp_id,dataset,nc,local_lr,pm_teacher_lr,pm_teacher_epochs,gpu_id,log_file" > "$MANIFEST"

echo "============================================================"
echo "FedCD Grid Search (base: time_194429)"
echo "GPU_ID=$GPU_ID, MAX_CONCURRENT=$MAX_CONCURRENT"
echo "GLOBAL_ROUNDS=$GLOBAL_ROUNDS, LOCAL_EPOCHS=$LOCAL_EPOCHS"
echo "SCENARIOS=${SCENARIOS[*]}, NCS=${NCS[*]}"
echo "LOCAL_LRS=${LOCAL_LRS[*]}"
echo "PM_TEACHER_LRS=${PM_TEACHER_LRS[*]}"
echo "PM_TEACHER_EPOCHS=${PM_TEACHER_EPOCHS[*]}"
echo "RUN_ROOT=$RUN_ROOT"
echo "============================================================"

exp_idx=0
running_jobs=0

build_dataset_name() {
    local scenario="$1"
    local nc="$2"
    if [ "$scenario" = "pat" ]; then
        echo "Cifar10_pat_nc${nc}"
    else
        local alpha="${scenario#dir}"
        echo "Cifar10_dir${alpha}_nc${nc}"
    fi
}

launch_one() {
    local dataset="$1"
    local nc="$2"
    local local_lr="$3"
    local pm_t_lr="$4"
    local pm_t_ep="$5"
    local exp_id="$6"

    local avg_data_per_client=$((TOTAL_DATA / nc))
    local cluster_sample_size=512
    if [ "$avg_data_per_client" -lt 512 ]; then
        cluster_sample_size="$avg_data_per_client"
    fi

    local log_file="$RUN_ROOT/${exp_id}.log"

    echo "${exp_id},${dataset},${nc},${local_lr},${pm_t_lr},${pm_t_ep},${GPU_ID},${log_file}" >> "$MANIFEST"
    echo "[Launch] $exp_id | $dataset | lr=$local_lr | pm_t_lr=$pm_t_lr | pm_t_ep=$pm_t_ep"

    (
        CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_CMD" -u system/main.py \
            -data "$dataset" \
            -algo FedCD \
            --gm_model VGG8 \
            --pm_model VGG8 \
            --fext_model SmallFExt \
            --fext_dim 512 \
            -gr "$GLOBAL_ROUNDS" \
            -nc "$nc" \
            -jr 1.0 \
            -lr "$local_lr" \
            --local_epochs "$LOCAL_EPOCHS" \
            --cluster_threshold 0.1 \
            --fedcd_enable_clustering True \
            --fedcd_enable_pm_aggregation True \
            --adaptive_threshold True \
            --threshold_step 0.01 \
            --threshold_step_max 0.1 \
            --threshold_decay 0.9 \
            --act_window_size 5 \
            --cluster_period 2 \
            --pm_period 1 \
            --global_period 2 \
            --cluster_sample_size "$cluster_sample_size" \
            --max_dynamic_clusters 0 \
            -dev cuda \
            -did 0 \
            -nw 0 \
            --pin_memory True \
            --prefetch_factor 2 \
            --amp True \
            --tf32 True \
            --gpu_batch_mult 1 \
            --gpu_batch_max 0 \
            --log_usage True \
            --avoid_oom True \
            --eval_common_global True \
            --global_test_samples 0 \
            --common_eval_batch_size 256 \
            --fedcd_local_pm_only_objective True \
            --fedcd_gm_update_mode server_pm_teacher \
            --fedcd_pm_teacher_lr "$pm_t_lr" \
            --fedcd_pm_teacher_temp "$PM_TEACHER_TEMP" \
            --fedcd_pm_teacher_kl_weight "$PM_TEACHER_KL_WEIGHT" \
            --fedcd_pm_teacher_ce_weight "$PM_TEACHER_CE_WEIGHT" \
            --fedcd_pm_teacher_epochs "$pm_t_ep" \
            --fedcd_pm_teacher_samples "$PM_TEACHER_SAMPLES" \
            --fedcd_pm_teacher_batch_size "$PM_TEACHER_BATCH_SIZE" \
            --fedcd_pm_teacher_proxy_dataset Cifar100 \
            --fedcd_pm_teacher_proxy_split train \
            --fedcd_pm_teacher_proxy_download False \
            --fedcd_pm_teacher_allow_test_fallback False \
            --fedcd_pm_teacher_confidence_weight True \
            --fedcd_pm_teacher_confidence_min "$PM_TEACHER_CONF_MIN" \
            --fedcd_pm_teacher_confidence_power "$PM_TEACHER_CONF_POWER" \
            --fedcd_pm_teacher_ensemble_confidence True \
            --fedcd_pm_teacher_topk "$PM_TEACHER_TOPK" \
            --fedcd_pm_teacher_abstain_threshold "$PM_TEACHER_ABSTAIN" \
            --fedcd_pm_teacher_rel_weight "$PM_TEACHER_REL_WEIGHT" \
            --fedcd_pm_teacher_rel_batch "$PM_TEACHER_REL_BATCH" \
            --fedcd_init_pretrain True \
            --fedcd_init_epochs 5 \
            --fedcd_init_lr 0.005 \
            --fedcd_init_samples 50000 \
            --fedcd_init_batch_size 256 \
            --fedcd_init_ce_weight 1.0 \
            --fedcd_init_kd_weight 1.0 \
            --fedcd_init_entropy_weight 0.05 \
            --fedcd_init_diversity_weight 0.05 \
            --fedcd_search_enable False \
            > "$log_file" 2>&1
    ) &
}

for nc in "${NCS[@]}"; do
    for scenario in "${SCENARIOS[@]}"; do
        dataset_name="$(build_dataset_name "$scenario" "$nc")"
        for local_lr in "${LOCAL_LRS[@]}"; do
            for pm_t_lr in "${PM_TEACHER_LRS[@]}"; do
                for pm_t_ep in "${PM_TEACHER_EPOCHS[@]}"; do
                    exp_idx=$((exp_idx + 1))
                    exp_id="$(printf "exp_%03d_%s_nc%s_lr%s_tlr%s_tep%s" "$exp_idx" "$scenario" "$nc" "$local_lr" "$pm_t_lr" "$pm_t_ep")"
                    launch_one "$dataset_name" "$nc" "$local_lr" "$pm_t_lr" "$pm_t_ep" "$exp_id"
                    running_jobs=$((running_jobs + 1))
                    if [ "$running_jobs" -ge "$MAX_CONCURRENT" ]; then
                        wait -n || true
                        running_jobs=$((running_jobs - 1))
                    fi
                done
            done
        done
    done
done

wait

echo "============================================================"
echo "All grid jobs finished."
echo "Manifest: $MANIFEST"
echo "Run logs: $RUN_ROOT"
echo "============================================================"
