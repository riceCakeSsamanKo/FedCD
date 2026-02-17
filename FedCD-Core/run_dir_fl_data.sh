#!/bin/bash

# Continue on error (skip failed experiments)
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -d "$SCRIPT_DIR/../fl_data" ]; then
    FL_DATA_ROOT="$SCRIPT_DIR/../fl_data"
elif [ -d "$SCRIPT_DIR/../../fl_data" ]; then
    FL_DATA_ROOT="$SCRIPT_DIR/../../fl_data"
else
    FL_DATA_ROOT="$SCRIPT_DIR/../fl_data"
fi

# Settings
GPU_DEVICE="cuda"
GLOBAL_ROUNDS=100
ALGO="FedCD"
DATASET="Cifar10"
TOTAL_DATA=50000
AVOID_OOM=True
FEDCD_FUSION_WEIGHT=1.0
FEDCD_GM_LOGITS_WEIGHT=1.0
FEDCD_PM_LOGITS_WEIGHT=0.3
FEDCD_PM_ONLY_WEIGHT=0.8
FEDCD_GM_LR_SCALE=0.1
FEDCD_GM_UPDATE_MODE="server_pm_teacher"
FEDCD_PM_TEACHER_LR=0.01
FEDCD_PM_TEACHER_TEMP=2.0
FEDCD_PM_TEACHER_KL_WEIGHT=1.0
FEDCD_PM_TEACHER_CE_WEIGHT=0.0
FEDCD_PM_TEACHER_SAMPLES=2000
FEDCD_PM_TEACHER_BATCH_SIZE=256
FEDCD_PM_TEACHER_PROXY_DATASET="Cifar100"
FEDCD_PM_TEACHER_PROXY_ROOT=""
FEDCD_PM_TEACHER_PROXY_SPLIT="train"
FEDCD_PM_TEACHER_PROXY_DOWNLOAD=False
FEDCD_PM_TEACHER_ALLOW_TEST_FALLBACK=False
FEDCD_PM_TEACHER_CONFIDENCE_WEIGHT=True
FEDCD_PM_TEACHER_CONFIDENCE_MIN=0.05
FEDCD_PM_TEACHER_CONFIDENCE_POWER=1.0
FEDCD_ENTROPY_TEMP_PM=1.0
FEDCD_ENTROPY_TEMP_GM=1.0
FEDCD_ENTROPY_MIN_PM_WEIGHT=0.1
FEDCD_ENTROPY_MAX_PM_WEIGHT=0.9

# List of Dirichlet alpha values to test
ALPHAS=(1.0) # (0.1 0.5 1.0)
# List of distance thresholds for Agglomerative Clustering
THRESHOLDS=(0.1)
CLIENT_COUNTS=(20 50)

echo "============================================================"
echo "Starting Experiment Suite for FedCD (Adaptive Threshold - ACT)"
echo "Tested Alphas: ${ALPHAS[*]}"
echo "Initial Thresholds to Test: ${THRESHOLDS[*]}"
echo "Local Loss Weights: fusion=${FEDCD_FUSION_WEIGHT}, gm_logits=${FEDCD_GM_LOGITS_WEIGHT}, pm_logits=${FEDCD_PM_LOGITS_WEIGHT}, pm_only=${FEDCD_PM_ONLY_WEIGHT}, gm_lr_scale=${FEDCD_GM_LR_SCALE}"
echo "GM Update Mode: ${FEDCD_GM_UPDATE_MODE} (pm_teacher: lr=${FEDCD_PM_TEACHER_LR}, temp=${FEDCD_PM_TEACHER_TEMP}, kl=${FEDCD_PM_TEACHER_KL_WEIGHT}, ce=${FEDCD_PM_TEACHER_CE_WEIGHT}, samples=${FEDCD_PM_TEACHER_SAMPLES}, batch=${FEDCD_PM_TEACHER_BATCH_SIZE}, proxy=${FEDCD_PM_TEACHER_PROXY_DATASET}/${FEDCD_PM_TEACHER_PROXY_SPLIT}, conf_w=${FEDCD_PM_TEACHER_CONFIDENCE_WEIGHT})"
echo "Entropy Gate: temp_pm=${FEDCD_ENTROPY_TEMP_PM}, temp_gm=${FEDCD_ENTROPY_TEMP_GM}, pm_range=[${FEDCD_ENTROPY_MIN_PM_WEIGHT},${FEDCD_ENTROPY_MAX_PM_WEIGHT}]"
echo "============================================================"

for ALPHA in "${ALPHAS[@]}"
do
    for THRESHOLD in "${THRESHOLDS[@]}"
    do
        for NUM_CLIENTS in "${CLIENT_COUNTS[@]}"
        do
            # Calculate safe cluster_sample_size
            AVG_DATA_PER_CLIENT=$((TOTAL_DATA / NUM_CLIENTS))
            if [ "$AVG_DATA_PER_CLIENT" -lt 512 ]; then
                CLUSTER_SAMPLE_SIZE=$AVG_DATA_PER_CLIENT
            else
                CLUSTER_SAMPLE_SIZE=512
            fi
            
            echo ""
            echo "############################################################"
            echo "ALPHA = $ALPHA"
            echo "NUM_CLIENTS = $NUM_CLIENTS"
            echo "CLUSTER_THRESHOLD = $THRESHOLD"
            echo "Adjusted cluster_sample_size = $CLUSTER_SAMPLE_SIZE"
            echo "############################################################"

            echo ""
            echo ">>> [Exp] Using Dirichlet (dir) | Alpha: $ALPHA | Clients: $NUM_CLIENTS"
            DATASET_NAME="Cifar10_dir${ALPHA}_nc${NUM_CLIENTS}"

            echo "Running Training (dir)..."
            START_TIME=$SECONDS
            python system/main.py \
                -data $DATASET_NAME \
                -algo $ALGO \
                --gm_model VGG16 \
                --pm_model VGG8 \
                -gr $GLOBAL_ROUNDS \
                -nc $NUM_CLIENTS \
                --cluster_threshold $THRESHOLD \
                --adaptive_threshold True \
                --threshold_step 0.05 \
                --threshold_decay 0.9 \
                --act_window_size 5 \
                --cluster_period 2 \
                --pm_period 1 \
                --global_period 2 \
                --cluster_sample_size $CLUSTER_SAMPLE_SIZE \
                --max_dynamic_clusters 0 \
                -dev $GPU_DEVICE \
                -nw 0 \
                --pin_memory True \
                --prefetch_factor 2 \
                --amp True \
                --tf32 True \
                --gpu_batch_mult 1 \
                --gpu_batch_max 0 \
                --log_usage True \
                --avoid_oom $AVOID_OOM \
                --fedcd_fusion_weight $FEDCD_FUSION_WEIGHT \
                --fedcd_gm_logits_weight $FEDCD_GM_LOGITS_WEIGHT \
                --fedcd_pm_logits_weight $FEDCD_PM_LOGITS_WEIGHT \
                --fedcd_pm_only_weight $FEDCD_PM_ONLY_WEIGHT \
                --fedcd_gm_lr_scale $FEDCD_GM_LR_SCALE \
                --fedcd_gm_update_mode $FEDCD_GM_UPDATE_MODE \
                --fedcd_pm_teacher_lr $FEDCD_PM_TEACHER_LR \
                --fedcd_pm_teacher_temp $FEDCD_PM_TEACHER_TEMP \
                --fedcd_pm_teacher_kl_weight $FEDCD_PM_TEACHER_KL_WEIGHT \
                --fedcd_pm_teacher_ce_weight $FEDCD_PM_TEACHER_CE_WEIGHT \
                --fedcd_pm_teacher_samples $FEDCD_PM_TEACHER_SAMPLES \
                --fedcd_pm_teacher_batch_size $FEDCD_PM_TEACHER_BATCH_SIZE \
                --fedcd_pm_teacher_proxy_dataset $FEDCD_PM_TEACHER_PROXY_DATASET \
                --fedcd_pm_teacher_proxy_root "$FEDCD_PM_TEACHER_PROXY_ROOT" \
                --fedcd_pm_teacher_proxy_split $FEDCD_PM_TEACHER_PROXY_SPLIT \
                --fedcd_pm_teacher_proxy_download $FEDCD_PM_TEACHER_PROXY_DOWNLOAD \
                --fedcd_pm_teacher_allow_test_fallback $FEDCD_PM_TEACHER_ALLOW_TEST_FALLBACK \
                --fedcd_pm_teacher_confidence_weight $FEDCD_PM_TEACHER_CONFIDENCE_WEIGHT \
                --fedcd_pm_teacher_confidence_min $FEDCD_PM_TEACHER_CONFIDENCE_MIN \
                --fedcd_pm_teacher_confidence_power $FEDCD_PM_TEACHER_CONFIDENCE_POWER \
                --fedcd_entropy_temp_pm $FEDCD_ENTROPY_TEMP_PM \
                --fedcd_entropy_temp_gm $FEDCD_ENTROPY_TEMP_GM \
                --fedcd_entropy_min_pm_weight $FEDCD_ENTROPY_MIN_PM_WEIGHT \
                --fedcd_entropy_max_pm_weight $FEDCD_ENTROPY_MAX_PM_WEIGHT \
                --local_epochs 1 || echo "Warning: Training (dir) failed for $NUM_CLIENTS clients. Skipping..."
            ELAPSED_TIME=$(($SECONDS - $START_TIME))

            # Copy dataset config to the latest log directory from fl_data
            LATEST_LOG_DIR=$(find logs -type d -name "time_*" | xargs ls -td | head -n 1)
            if [ -d "$LATEST_LOG_DIR" ]; then
                cp "$FL_DATA_ROOT/$DATASET_NAME/config.json" "$LATEST_LOG_DIR/dataset_config_dir_ALPHA_${ALPHA}_THRESHOLD_${THRESHOLD}_NUM_CLIENTS_${NUM_CLIENTS}.json"
                echo "Dirichlet (dir) execution time: ${ELAPSED_TIME}s" >> "$LATEST_LOG_DIR/time.txt"
                echo "[Shell] Copied dataset config from fl_data to $LATEST_LOG_DIR"
            fi

            echo ">>> Exp 2 (dir) Finished."
            sleep 5
        done
    done
done
