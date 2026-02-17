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
LOCAL_EPOCHS=5
ALGO="FedCD"
DATASET="Cifar10"
TOTAL_DATA=50000
AVOID_OOM=True
# Global test evaluation settings
EVAL_COMMON_GLOBAL=True
GLOBAL_TEST_SAMPLES=0
COMMON_EVAL_BATCH_SIZE=256
FEDCD_ENABLE_CLUSTERING=True
FEDCD_GM_MODEL="VGG8"
FEDCD_PM_MODEL="VGG8"

FEDCD_FUSION_WEIGHT=0.6
FEDCD_GM_LOGITS_WEIGHT=0.0
FEDCD_PM_LOGITS_WEIGHT=0.7
FEDCD_PM_ONLY_WEIGHT=1.5
FEDCD_GM_LR_SCALE=0.1
FEDCD_GLOBAL_PERIOD=2
FEDCD_GM_UPDATE_MODE="server_pm_teacher"
FEDCD_HYBRID_PROTO_BLEND=0.35
FEDCD_NC_WEIGHT=0.02
FEDCD_NC_TARGET_CORR=-0.1
FEDCD_PM_TEACHER_LR=0.01
FEDCD_PM_TEACHER_TEMP=2.0
FEDCD_PM_TEACHER_KL_WEIGHT=1.0
FEDCD_PM_TEACHER_CE_WEIGHT=0.0
FEDCD_PM_TEACHER_EPOCHS=3
FEDCD_PM_TEACHER_SAMPLES=50000
FEDCD_PM_TEACHER_BATCH_SIZE=256
FEDCD_PM_TEACHER_PROXY_DATASET="Cifar100"
FEDCD_PM_TEACHER_PROXY_ROOT=""
FEDCD_PM_TEACHER_PROXY_SPLIT="train"
FEDCD_PM_TEACHER_PROXY_DOWNLOAD=False
FEDCD_PM_TEACHER_ALLOW_TEST_FALLBACK=False
FEDCD_PM_TEACHER_CONFIDENCE_WEIGHT=True
FEDCD_PM_TEACHER_CONFIDENCE_MIN=0.05
FEDCD_PM_TEACHER_CONFIDENCE_POWER=2.0
FEDCD_PM_TEACHER_ENSEMBLE_CONFIDENCE=True
FEDCD_PM_TEACHER_TOPK=4
FEDCD_PM_TEACHER_ABSTAIN_THRESHOLD=0.2
FEDCD_PM_TEACHER_REL_WEIGHT=0.0
FEDCD_PM_TEACHER_REL_BATCH=64
FEDCD_INIT_PRETRAIN=True
FEDCD_INIT_EPOCHS=5
FEDCD_INIT_LR=0.005
FEDCD_INIT_SAMPLES=50000
FEDCD_INIT_BATCH_SIZE=256
FEDCD_INIT_CE_WEIGHT=1.0
FEDCD_INIT_KD_WEIGHT=1.0
FEDCD_INIT_ENTROPY_WEIGHT=0.05
FEDCD_INIT_DIVERSITY_WEIGHT=0.05
FEDCD_PROTO_TEACHER_LR=0.005
FEDCD_PROTO_TEACHER_STEPS=120
FEDCD_PROTO_TEACHER_BATCH_SIZE=256
FEDCD_PROTO_TEACHER_TEMP=2.0
FEDCD_PROTO_TEACHER_CE_WEIGHT=1.0
FEDCD_PROTO_TEACHER_KL_WEIGHT=0.5
FEDCD_PROTO_TEACHER_NOISE_SCALE=1.0
FEDCD_PROTO_TEACHER_MIN_COUNT=1
FEDCD_PROTO_TEACHER_CLIENT_SAMPLES=0
FEDCD_PROTO_TEACHER_CONFIDENCE_WEIGHT=True
FEDCD_PROTO_TEACHER_CONFIDENCE_MIN=0.05
FEDCD_PROTO_TEACHER_CONFIDENCE_POWER=1.0
FEDCD_ENTROPY_TEMP_PM=1.0
FEDCD_ENTROPY_TEMP_GM=1.0
FEDCD_ENTROPY_MIN_PM_WEIGHT=0.5
FEDCD_ENTROPY_MAX_PM_WEIGHT=0.5
FEDCD_ENTROPY_GATE_TAU=0.15
FEDCD_ENTROPY_PM_BIAS=0.0
FEDCD_ENTROPY_GM_BIAS=0.0
FEDCD_ENTROPY_DISAGREE_GM_BOOST=0.0
FEDCD_ENTROPY_USE_CLASS_RELIABILITY=False
FEDCD_ENTROPY_RELIABILITY_SCALE=0.8
FEDCD_ENTROPY_HARD_SWITCH_MARGIN=0.35
FEDCD_ENTROPY_USE_OOD_GATE=True
FEDCD_ENTROPY_OOD_SCALE=1.0
FEDCD_GATE_RELIABILITY_EMA=0.9
FEDCD_GATE_RELIABILITY_SAMPLES=512
FEDCD_GATE_FEATURE_EMA=0.9
FEDCD_GATE_FEATURE_SAMPLES=512

# List of Dirichlet alpha values to test
ALPHAS=(0.1 0.5 1.0) # (0.1 0.5 1.0)
# List of distance thresholds for Agglomerative Clustering
THRESHOLDS=(0.0)
CLIENT_COUNTS=(20 50)

echo "============================================================"
echo "Starting Experiment Suite for FedCD (Adaptive Threshold - ACT)"
echo "Tested Alphas: ${ALPHAS[*]}"
echo "Initial Thresholds to Test: ${THRESHOLDS[*]}"
echo "Global Test Eval: ${EVAL_COMMON_GLOBAL} (samples=${GLOBAL_TEST_SAMPLES}, batch=${COMMON_EVAL_BATCH_SIZE})"
echo "Clustering Enabled: ${FEDCD_ENABLE_CLUSTERING}"
echo "Model Setup: GM=${FEDCD_GM_MODEL}, PM=${FEDCD_PM_MODEL}, FExt=SmallFExt"
echo "Local Loss Weights: fusion=${FEDCD_FUSION_WEIGHT}, gm_logits=${FEDCD_GM_LOGITS_WEIGHT}, pm_logits=${FEDCD_PM_LOGITS_WEIGHT}, pm_only=${FEDCD_PM_ONLY_WEIGHT}, gm_lr_scale=${FEDCD_GM_LR_SCALE}, nc_w=${FEDCD_NC_WEIGHT}, nc_target=${FEDCD_NC_TARGET_CORR}"
echo "GM Update Mode: ${FEDCD_GM_UPDATE_MODE} (hybrid_blend=${FEDCD_HYBRID_PROTO_BLEND}; pm_teacher: epochs=${FEDCD_PM_TEACHER_EPOCHS}, lr=${FEDCD_PM_TEACHER_LR}, temp=${FEDCD_PM_TEACHER_TEMP}, kl=${FEDCD_PM_TEACHER_KL_WEIGHT}, ce=${FEDCD_PM_TEACHER_CE_WEIGHT}, rel=${FEDCD_PM_TEACHER_REL_WEIGHT}, rel_batch=${FEDCD_PM_TEACHER_REL_BATCH}, samples=${FEDCD_PM_TEACHER_SAMPLES}, batch=${FEDCD_PM_TEACHER_BATCH_SIZE}, proxy=${FEDCD_PM_TEACHER_PROXY_DATASET}/${FEDCD_PM_TEACHER_PROXY_SPLIT}, conf_w=${FEDCD_PM_TEACHER_CONFIDENCE_WEIGHT}, conf_min=${FEDCD_PM_TEACHER_CONFIDENCE_MIN}, conf_pow=${FEDCD_PM_TEACHER_CONFIDENCE_POWER}, ens_conf=${FEDCD_PM_TEACHER_ENSEMBLE_CONFIDENCE}, topk=${FEDCD_PM_TEACHER_TOPK}, abstain=${FEDCD_PM_TEACHER_ABSTAIN_THRESHOLD})"
echo "Init Pretrain: enabled=${FEDCD_INIT_PRETRAIN}, epochs=${FEDCD_INIT_EPOCHS}, lr=${FEDCD_INIT_LR}, samples=${FEDCD_INIT_SAMPLES}, batch=${FEDCD_INIT_BATCH_SIZE}, ce=${FEDCD_INIT_CE_WEIGHT}, kd=${FEDCD_INIT_KD_WEIGHT}, ent=${FEDCD_INIT_ENTROPY_WEIGHT}, div=${FEDCD_INIT_DIVERSITY_WEIGHT}"
echo "Prototype Teacher: lr=${FEDCD_PROTO_TEACHER_LR}, steps=${FEDCD_PROTO_TEACHER_STEPS}, batch=${FEDCD_PROTO_TEACHER_BATCH_SIZE}, temp=${FEDCD_PROTO_TEACHER_TEMP}, ce=${FEDCD_PROTO_TEACHER_CE_WEIGHT}, kl=${FEDCD_PROTO_TEACHER_KL_WEIGHT}, noise=${FEDCD_PROTO_TEACHER_NOISE_SCALE}, min_count=${FEDCD_PROTO_TEACHER_MIN_COUNT}, client_samples=${FEDCD_PROTO_TEACHER_CLIENT_SAMPLES}, conf_w=${FEDCD_PROTO_TEACHER_CONFIDENCE_WEIGHT}"
echo "Entropy Gate: temp_pm=${FEDCD_ENTROPY_TEMP_PM}, temp_gm=${FEDCD_ENTROPY_TEMP_GM}, pm_range=[${FEDCD_ENTROPY_MIN_PM_WEIGHT},${FEDCD_ENTROPY_MAX_PM_WEIGHT}], gate_tau=${FEDCD_ENTROPY_GATE_TAU}, pm_bias=${FEDCD_ENTROPY_PM_BIAS}, gm_bias=${FEDCD_ENTROPY_GM_BIAS}, disagree_gm_boost=${FEDCD_ENTROPY_DISAGREE_GM_BOOST}, class_rel=${FEDCD_ENTROPY_USE_CLASS_RELIABILITY}, rel_scale=${FEDCD_ENTROPY_RELIABILITY_SCALE}, hard_margin=${FEDCD_ENTROPY_HARD_SWITCH_MARGIN}, ood_gate=${FEDCD_ENTROPY_USE_OOD_GATE}, ood_scale=${FEDCD_ENTROPY_OOD_SCALE}"
echo "============================================================"

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
            echo "NUM_CLIENTS = $NUM_CLIENTS"
            echo "CLUSTER_THRESHOLD = $THRESHOLD"
            echo "Adjusted cluster_sample_size = $CLUSTER_SAMPLE_SIZE"
            echo "############################################################"

            # ------------------------------------------------------------------
            # Experiment 1: Pathological Non-IID (pat) - Balanced
            # ------------------------------------------------------------------
            DATASET_NAME="Cifar10_pat_nc${NUM_CLIENTS}"
            echo ""
            echo ">>> [Exp 1/2] Using Pathological (pat) | Clients: $NUM_CLIENTS"
            
            echo "Running Training (pat)..."
            START_TIME=$SECONDS
            python system/main.py \
                -data $DATASET_NAME \
                -algo $ALGO \
                --gm_model $FEDCD_GM_MODEL \
                --pm_model $FEDCD_PM_MODEL \
                --fext_model SmallFExt \
                --fext_dim 512 \
                -gr $GLOBAL_ROUNDS \
                -nc $NUM_CLIENTS \
                --cluster_threshold $THRESHOLD \
                --fedcd_enable_clustering $FEDCD_ENABLE_CLUSTERING \
                --adaptive_threshold True \
                --threshold_step 0.01 \
                --threshold_step_max 0.1 \
                --threshold_decay 0.9 \
                --act_window_size 5 \
                --cluster_period 2 \
                --pm_period 1 \
                --global_period $FEDCD_GLOBAL_PERIOD \
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
                --eval_common_global $EVAL_COMMON_GLOBAL \
                --global_test_samples $GLOBAL_TEST_SAMPLES \
                --common_eval_batch_size $COMMON_EVAL_BATCH_SIZE \
                --fedcd_fusion_weight $FEDCD_FUSION_WEIGHT \
                --fedcd_nc_weight $FEDCD_NC_WEIGHT \
                --fedcd_nc_target_corr $FEDCD_NC_TARGET_CORR \
                --fedcd_gm_logits_weight $FEDCD_GM_LOGITS_WEIGHT \
                --fedcd_pm_logits_weight $FEDCD_PM_LOGITS_WEIGHT \
                --fedcd_pm_only_weight $FEDCD_PM_ONLY_WEIGHT \
                --fedcd_gm_lr_scale $FEDCD_GM_LR_SCALE \
                --fedcd_gm_update_mode $FEDCD_GM_UPDATE_MODE \
                --fedcd_hybrid_proto_blend $FEDCD_HYBRID_PROTO_BLEND \
                --fedcd_pm_teacher_lr $FEDCD_PM_TEACHER_LR \
                --fedcd_pm_teacher_temp $FEDCD_PM_TEACHER_TEMP \
                --fedcd_pm_teacher_kl_weight $FEDCD_PM_TEACHER_KL_WEIGHT \
                --fedcd_pm_teacher_ce_weight $FEDCD_PM_TEACHER_CE_WEIGHT \
                --fedcd_pm_teacher_epochs $FEDCD_PM_TEACHER_EPOCHS \
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
                --fedcd_pm_teacher_ensemble_confidence $FEDCD_PM_TEACHER_ENSEMBLE_CONFIDENCE \
                --fedcd_pm_teacher_topk $FEDCD_PM_TEACHER_TOPK \
                --fedcd_pm_teacher_abstain_threshold $FEDCD_PM_TEACHER_ABSTAIN_THRESHOLD \
                --fedcd_pm_teacher_rel_weight $FEDCD_PM_TEACHER_REL_WEIGHT \
                --fedcd_pm_teacher_rel_batch $FEDCD_PM_TEACHER_REL_BATCH \
                --fedcd_init_pretrain $FEDCD_INIT_PRETRAIN \
                --fedcd_init_epochs $FEDCD_INIT_EPOCHS \
                --fedcd_init_lr $FEDCD_INIT_LR \
                --fedcd_init_samples $FEDCD_INIT_SAMPLES \
                --fedcd_init_batch_size $FEDCD_INIT_BATCH_SIZE \
                --fedcd_init_ce_weight $FEDCD_INIT_CE_WEIGHT \
                --fedcd_init_kd_weight $FEDCD_INIT_KD_WEIGHT \
                --fedcd_init_entropy_weight $FEDCD_INIT_ENTROPY_WEIGHT \
                --fedcd_init_diversity_weight $FEDCD_INIT_DIVERSITY_WEIGHT \
                --fedcd_proto_teacher_lr $FEDCD_PROTO_TEACHER_LR \
                --fedcd_proto_teacher_steps $FEDCD_PROTO_TEACHER_STEPS \
                --fedcd_proto_teacher_batch_size $FEDCD_PROTO_TEACHER_BATCH_SIZE \
                --fedcd_proto_teacher_temp $FEDCD_PROTO_TEACHER_TEMP \
                --fedcd_proto_teacher_ce_weight $FEDCD_PROTO_TEACHER_CE_WEIGHT \
                --fedcd_proto_teacher_kl_weight $FEDCD_PROTO_TEACHER_KL_WEIGHT \
                --fedcd_proto_teacher_noise_scale $FEDCD_PROTO_TEACHER_NOISE_SCALE \
                --fedcd_proto_teacher_min_count $FEDCD_PROTO_TEACHER_MIN_COUNT \
                --fedcd_proto_teacher_client_samples $FEDCD_PROTO_TEACHER_CLIENT_SAMPLES \
                --fedcd_proto_teacher_confidence_weight $FEDCD_PROTO_TEACHER_CONFIDENCE_WEIGHT \
                --fedcd_proto_teacher_confidence_min $FEDCD_PROTO_TEACHER_CONFIDENCE_MIN \
                --fedcd_proto_teacher_confidence_power $FEDCD_PROTO_TEACHER_CONFIDENCE_POWER \
                --fedcd_entropy_temp_pm $FEDCD_ENTROPY_TEMP_PM \
                --fedcd_entropy_temp_gm $FEDCD_ENTROPY_TEMP_GM \
                --fedcd_entropy_min_pm_weight $FEDCD_ENTROPY_MIN_PM_WEIGHT \
                --fedcd_entropy_max_pm_weight $FEDCD_ENTROPY_MAX_PM_WEIGHT \
                --fedcd_entropy_gate_tau $FEDCD_ENTROPY_GATE_TAU \
                --fedcd_entropy_pm_bias $FEDCD_ENTROPY_PM_BIAS \
                --fedcd_entropy_gm_bias $FEDCD_ENTROPY_GM_BIAS \
                --fedcd_entropy_disagree_gm_boost $FEDCD_ENTROPY_DISAGREE_GM_BOOST \
                --fedcd_entropy_use_class_reliability $FEDCD_ENTROPY_USE_CLASS_RELIABILITY \
                --fedcd_entropy_reliability_scale $FEDCD_ENTROPY_RELIABILITY_SCALE \
                --fedcd_entropy_hard_switch_margin $FEDCD_ENTROPY_HARD_SWITCH_MARGIN \
                --fedcd_entropy_use_ood_gate $FEDCD_ENTROPY_USE_OOD_GATE \
                --fedcd_entropy_ood_scale $FEDCD_ENTROPY_OOD_SCALE \
                --fedcd_gate_reliability_ema $FEDCD_GATE_RELIABILITY_EMA \
                --fedcd_gate_reliability_samples $FEDCD_GATE_RELIABILITY_SAMPLES \
                --fedcd_gate_feature_ema $FEDCD_GATE_FEATURE_EMA \
                --fedcd_gate_feature_samples $FEDCD_GATE_FEATURE_SAMPLES \
                --local_epochs $LOCAL_EPOCHS || echo "Warning: Training (pat) failed for $NUM_CLIENTS clients. Skipping..."
            ELAPSED_TIME=$(($SECONDS - $START_TIME))

            # Copy dataset config to the latest log directory from fl_data
            LATEST_LOG_DIR=$(find logs -type d -name "time_*" | xargs ls -td | head -n 1)
            if [ -d "$LATEST_LOG_DIR" ]; then
                cp "$FL_DATA_ROOT/$DATASET_NAME/config.json" "$LATEST_LOG_DIR/dataset_config_pat_THRESHOLD_${THRESHOLD}_NUM_CLIENTS_${NUM_CLIENTS}.json"
                echo "Pathological (pat) execution time: ${ELAPSED_TIME}s" >> "$LATEST_LOG_DIR/time.txt"
                echo "[Shell] Copied dataset config from fl_data to $LATEST_LOG_DIR"
            fi

            echo ">>> Exp 1 (pat) Finished."
            sleep 5

            for ALPHA in "${ALPHAS[@]}"
            do
                echo ""
                echo "------------------------------------------------------------"
                echo "Running Dirichlet for ALPHA = $ALPHA"
                echo "------------------------------------------------------------"

                # ------------------------------------------------------------------
                # Experiment 2: Dirichlet Non-IID (dir) - Unbalanced
                # ------------------------------------------------------------------
                DATASET_NAME="Cifar10_dir${ALPHA}_nc${NUM_CLIENTS}"
                echo ""
                echo ">>> [Exp 2/2] Using Dirichlet (dir) | Alpha: $ALPHA | Clients: $NUM_CLIENTS"

                echo "Running Training (dir)..."
                START_TIME=$SECONDS
                python system/main.py \
                    -data $DATASET_NAME \
                    -algo $ALGO \
                    --gm_model $FEDCD_GM_MODEL \
                    --pm_model $FEDCD_PM_MODEL \
                    --fext_model SmallFExt \
                    --fext_dim 512 \
                    -gr $GLOBAL_ROUNDS \
                    -nc $NUM_CLIENTS \
                    --cluster_threshold $THRESHOLD \
                    --fedcd_enable_clustering $FEDCD_ENABLE_CLUSTERING \
                    --adaptive_threshold True \
                    --threshold_step 0.01 \
                    --threshold_step_max 0.1 \
                    --threshold_decay 0.9 \
                    --act_window_size 5 \
                    --cluster_period 2 \
                    --pm_period 1 \
                    --global_period $FEDCD_GLOBAL_PERIOD \
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
                    --eval_common_global $EVAL_COMMON_GLOBAL \
                    --global_test_samples $GLOBAL_TEST_SAMPLES \
                    --common_eval_batch_size $COMMON_EVAL_BATCH_SIZE \
                    --fedcd_fusion_weight $FEDCD_FUSION_WEIGHT \
                    --fedcd_nc_weight $FEDCD_NC_WEIGHT \
                    --fedcd_nc_target_corr $FEDCD_NC_TARGET_CORR \
                    --fedcd_gm_logits_weight $FEDCD_GM_LOGITS_WEIGHT \
                    --fedcd_pm_logits_weight $FEDCD_PM_LOGITS_WEIGHT \
                    --fedcd_pm_only_weight $FEDCD_PM_ONLY_WEIGHT \
                    --fedcd_gm_lr_scale $FEDCD_GM_LR_SCALE \
                    --fedcd_gm_update_mode $FEDCD_GM_UPDATE_MODE \
                    --fedcd_hybrid_proto_blend $FEDCD_HYBRID_PROTO_BLEND \
                    --fedcd_pm_teacher_lr $FEDCD_PM_TEACHER_LR \
                    --fedcd_pm_teacher_temp $FEDCD_PM_TEACHER_TEMP \
                    --fedcd_pm_teacher_kl_weight $FEDCD_PM_TEACHER_KL_WEIGHT \
                    --fedcd_pm_teacher_ce_weight $FEDCD_PM_TEACHER_CE_WEIGHT \
                    --fedcd_pm_teacher_epochs $FEDCD_PM_TEACHER_EPOCHS \
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
                    --fedcd_pm_teacher_ensemble_confidence $FEDCD_PM_TEACHER_ENSEMBLE_CONFIDENCE \
                    --fedcd_pm_teacher_topk $FEDCD_PM_TEACHER_TOPK \
                    --fedcd_pm_teacher_abstain_threshold $FEDCD_PM_TEACHER_ABSTAIN_THRESHOLD \
                    --fedcd_pm_teacher_rel_weight $FEDCD_PM_TEACHER_REL_WEIGHT \
                    --fedcd_pm_teacher_rel_batch $FEDCD_PM_TEACHER_REL_BATCH \
                    --fedcd_init_pretrain $FEDCD_INIT_PRETRAIN \
                    --fedcd_init_epochs $FEDCD_INIT_EPOCHS \
                    --fedcd_init_lr $FEDCD_INIT_LR \
                    --fedcd_init_samples $FEDCD_INIT_SAMPLES \
                    --fedcd_init_batch_size $FEDCD_INIT_BATCH_SIZE \
                    --fedcd_init_ce_weight $FEDCD_INIT_CE_WEIGHT \
                    --fedcd_init_kd_weight $FEDCD_INIT_KD_WEIGHT \
                    --fedcd_init_entropy_weight $FEDCD_INIT_ENTROPY_WEIGHT \
                    --fedcd_init_diversity_weight $FEDCD_INIT_DIVERSITY_WEIGHT \
                    --fedcd_proto_teacher_lr $FEDCD_PROTO_TEACHER_LR \
                    --fedcd_proto_teacher_steps $FEDCD_PROTO_TEACHER_STEPS \
                    --fedcd_proto_teacher_batch_size $FEDCD_PROTO_TEACHER_BATCH_SIZE \
                    --fedcd_proto_teacher_temp $FEDCD_PROTO_TEACHER_TEMP \
                    --fedcd_proto_teacher_ce_weight $FEDCD_PROTO_TEACHER_CE_WEIGHT \
                    --fedcd_proto_teacher_kl_weight $FEDCD_PROTO_TEACHER_KL_WEIGHT \
                    --fedcd_proto_teacher_noise_scale $FEDCD_PROTO_TEACHER_NOISE_SCALE \
                    --fedcd_proto_teacher_min_count $FEDCD_PROTO_TEACHER_MIN_COUNT \
                    --fedcd_proto_teacher_client_samples $FEDCD_PROTO_TEACHER_CLIENT_SAMPLES \
                    --fedcd_proto_teacher_confidence_weight $FEDCD_PROTO_TEACHER_CONFIDENCE_WEIGHT \
                    --fedcd_proto_teacher_confidence_min $FEDCD_PROTO_TEACHER_CONFIDENCE_MIN \
                    --fedcd_proto_teacher_confidence_power $FEDCD_PROTO_TEACHER_CONFIDENCE_POWER \
                    --fedcd_entropy_temp_pm $FEDCD_ENTROPY_TEMP_PM \
                    --fedcd_entropy_temp_gm $FEDCD_ENTROPY_TEMP_GM \
                    --fedcd_entropy_min_pm_weight $FEDCD_ENTROPY_MIN_PM_WEIGHT \
                    --fedcd_entropy_max_pm_weight $FEDCD_ENTROPY_MAX_PM_WEIGHT \
                    --fedcd_entropy_gate_tau $FEDCD_ENTROPY_GATE_TAU \
                    --fedcd_entropy_pm_bias $FEDCD_ENTROPY_PM_BIAS \
                    --fedcd_entropy_gm_bias $FEDCD_ENTROPY_GM_BIAS \
                    --fedcd_entropy_disagree_gm_boost $FEDCD_ENTROPY_DISAGREE_GM_BOOST \
                    --fedcd_entropy_use_class_reliability $FEDCD_ENTROPY_USE_CLASS_RELIABILITY \
                    --fedcd_entropy_reliability_scale $FEDCD_ENTROPY_RELIABILITY_SCALE \
                    --fedcd_entropy_hard_switch_margin $FEDCD_ENTROPY_HARD_SWITCH_MARGIN \
                    --fedcd_entropy_use_ood_gate $FEDCD_ENTROPY_USE_OOD_GATE \
                    --fedcd_entropy_ood_scale $FEDCD_ENTROPY_OOD_SCALE \
                    --fedcd_gate_reliability_ema $FEDCD_GATE_RELIABILITY_EMA \
                    --fedcd_gate_reliability_samples $FEDCD_GATE_RELIABILITY_SAMPLES \
                    --fedcd_gate_feature_ema $FEDCD_GATE_FEATURE_EMA \
                    --fedcd_gate_feature_samples $FEDCD_GATE_FEATURE_SAMPLES \
                    --local_epochs $LOCAL_EPOCHS || echo "Warning: Training (dir) failed for $NUM_CLIENTS clients. Skipping..."
                ELAPSED_TIME=$(($SECONDS - $START_TIME))

                # Copy dataset config to the latest log directory from fl_data
                LATEST_LOG_DIR=$(find logs -type d -name "time_*" | xargs ls -td | head -n 1)
                if [ -d "$LATEST_LOG_DIR" ]; then
                    cp "$FL_DATA_ROOT/$DATASET_NAME/config.json" "$LATEST_LOG_DIR/dataset_config_dir_ALPHA_${ALPHA}_THRESHOLD_${THRESHOLD}_NUM_CLIENTS_${NUM_CLIENTS}.json"
                    echo "Dirichlet (dir) execution time: ${ELAPSED_TIME}s" >> "$LATEST_LOG_DIR/time.txt"
                    echo "[Shell] Copied dataset config from fl_data to $LATEST_LOG_DIR"
                fi
            done
            echo ">>> Exp 2 (dir) Finished for NUM_CLIENTS=$NUM_CLIENTS"
            sleep 5
        done
    done

echo "============================================================"
echo "All experiments completed."
echo "============================================================"
