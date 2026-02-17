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
FEDCD_DISTILL_LR=0.01
FEDCD_DISTILL_TEMP=3.0
FEDCD_DISTILL_KL_WEIGHT=1.0
FEDCD_DISTILL_CE_WEIGHT=0.05
FEDCD_FUSION_WEIGHT=1.0
FEDCD_GM_LOGITS_WEIGHT=1.0
FEDCD_PM_LOGITS_WEIGHT=0.3
FEDCD_PM_ONLY_WEIGHT=0.8
FEDCD_PROTOTYPE_SAMPLES=512
FEDCD_PROTO_WEIGHT=0.3
FEDCD_RELATION_WEIGHT=0.1
FEDCD_COMBINER_CALIB_EPOCHS=1
FEDCD_COMBINER_CALIB_LR_MULT=1.0

# List of Dirichlet alpha values to test
ALPHAS=(1.0) # (0.1 0.5 1.0)
# List of distance thresholds for Agglomerative Clustering
THRESHOLDS=(0.1)
CLIENT_COUNTS=(20 50)

echo "============================================================"
echo "Starting Experiment Suite for FedCD (Adaptive Threshold - ACT)"
echo "Tested Alphas: ${ALPHAS[*]}"
echo "Initial Thresholds to Test: ${THRESHOLDS[*]}"
echo "Distill: lr=${FEDCD_DISTILL_LR}, temp=${FEDCD_DISTILL_TEMP}, kl=${FEDCD_DISTILL_KL_WEIGHT}, ce=${FEDCD_DISTILL_CE_WEIGHT}"
echo "Local Loss Weights: fusion=${FEDCD_FUSION_WEIGHT}, gm_logits=${FEDCD_GM_LOGITS_WEIGHT}, pm_logits=${FEDCD_PM_LOGITS_WEIGHT}, pm_only=${FEDCD_PM_ONLY_WEIGHT}"
echo "Prototype Consensus: samples=${FEDCD_PROTOTYPE_SAMPLES}, proto_w=${FEDCD_PROTO_WEIGHT}, rel_w=${FEDCD_RELATION_WEIGHT}"
echo "Post-GM Combiner Calibration: epochs=${FEDCD_COMBINER_CALIB_EPOCHS}, lr_mult=${FEDCD_COMBINER_CALIB_LR_MULT}"
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
                --global_period 4 \
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
                --fedcd_distill_lr $FEDCD_DISTILL_LR \
                --fedcd_distill_temp $FEDCD_DISTILL_TEMP \
                --fedcd_distill_kl_weight $FEDCD_DISTILL_KL_WEIGHT \
                --fedcd_distill_ce_weight $FEDCD_DISTILL_CE_WEIGHT \
                --fedcd_fusion_weight $FEDCD_FUSION_WEIGHT \
                --fedcd_gm_logits_weight $FEDCD_GM_LOGITS_WEIGHT \
                --fedcd_pm_logits_weight $FEDCD_PM_LOGITS_WEIGHT \
                --fedcd_pm_only_weight $FEDCD_PM_ONLY_WEIGHT \
                --fedcd_prototype_samples $FEDCD_PROTOTYPE_SAMPLES \
                --fedcd_proto_weight $FEDCD_PROTO_WEIGHT \
                --fedcd_relation_weight $FEDCD_RELATION_WEIGHT \
                --fedcd_combiner_calib_epochs $FEDCD_COMBINER_CALIB_EPOCHS \
                --fedcd_combiner_calib_lr_mult $FEDCD_COMBINER_CALIB_LR_MULT \
                --broadcast_global_combiner False \
                --local_epochs 1 \
                --proxy_dataset Cifar100 --proxy_samples 2000 || echo "Warning: Training (dir) failed for $NUM_CLIENTS clients. Skipping..."
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
