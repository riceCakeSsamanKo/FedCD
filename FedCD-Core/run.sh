#!/usr/bin/env bash
set -euo pipefail

# Baseline-aligned runtime defaults
CONDA_ENV="${CONDA_ENV:-pfllib}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-Cifar10_pat_nc20}"
NUM_CLASSES="${NUM_CLASSES:-10}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
JOIN_RATIO="${JOIN_RATIO:-1.0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
LR="${LR:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GOAL="${GOAL:-train}"

# Fixed FedCD setting (reference: GM_VGG8W224_PM_VGG8W1322_Fext_SmallFExt)
GM_MODEL="${GM_MODEL:-VGG8W224}"
PM_MODEL="${PM_MODEL:-VGG8W1322}"
FEXT_MODEL="${FEXT_MODEL:-SmallFExt}"
FEXT_DIM="${FEXT_DIM:-256}"

# Teacher setup (client_conf)
TEACHER_PROXY_DATASET="${TEACHER_PROXY_DATASET:-Cifar100}"
TEACHER_SOURCE="${TEACHER_SOURCE:-client}"
TEACHER_TOPK="${TEACHER_TOPK:-5}"
TEACHER_ABSTAIN_THRESHOLD="${TEACHER_ABSTAIN_THRESHOLD:-0.3}"
TEACHER_TEACHER_ABSTAIN_THRESHOLD="${TEACHER_TEACHER_ABSTAIN_THRESHOLD:-0.5}"
TEACHER_MIN_ACTIVE="${TEACHER_MIN_ACTIVE:-2}"
TEACHER_CONSENSUS_MIN_RATIO="${TEACHER_CONSENSUS_MIN_RATIO:-0.6}"
FEDCD_PM_TEACHER_SAMPLES="${FEDCD_PM_TEACHER_SAMPLES:-50000}"
FEDCD_PM_TEACHER_BATCH_SIZE="${FEDCD_PM_TEACHER_BATCH_SIZE:-256}"
FEDCD_PM_TEACHER_EPOCHS="${FEDCD_PM_TEACHER_EPOCHS:-6}"
FEDCD_INIT_EPOCHS="${FEDCD_INIT_EPOCHS:-5}"
FEDCD_INIT_SAMPLES="${FEDCD_INIT_SAMPLES:-50000}"
FEDCD_INIT_BATCH_SIZE="${FEDCD_INIT_BATCH_SIZE:-256}"

# PM-first inference fusion defaults
FEDCD_ENTROPY_TEMP_PM="${FEDCD_ENTROPY_TEMP_PM:-1.0}"
FEDCD_ENTROPY_TEMP_GM="${FEDCD_ENTROPY_TEMP_GM:-1.0}"
FEDCD_ENTROPY_MIN_PM_WEIGHT="${FEDCD_ENTROPY_MIN_PM_WEIGHT:-0.80}"
FEDCD_ENTROPY_MAX_PM_WEIGHT="${FEDCD_ENTROPY_MAX_PM_WEIGHT:-0.98}"
FEDCD_ENTROPY_GATE_TAU="${FEDCD_ENTROPY_GATE_TAU:-0.10}"
FEDCD_ENTROPY_PM_BIAS="${FEDCD_ENTROPY_PM_BIAS:-0.05}"
FEDCD_ENTROPY_GM_BIAS="${FEDCD_ENTROPY_GM_BIAS:-0.0}"
FEDCD_ENTROPY_DISAGREE_GM_BOOST="${FEDCD_ENTROPY_DISAGREE_GM_BOOST:-0.0}"
FEDCD_ENTROPY_USE_CLASS_RELIABILITY="${FEDCD_ENTROPY_USE_CLASS_RELIABILITY:-true}"
FEDCD_ENTROPY_RELIABILITY_SCALE="${FEDCD_ENTROPY_RELIABILITY_SCALE:-0.8}"
FEDCD_ENTROPY_HARD_SWITCH_MARGIN="${FEDCD_ENTROPY_HARD_SWITCH_MARGIN:-0.20}"
FEDCD_ENTROPY_USE_OOD_GATE="${FEDCD_ENTROPY_USE_OOD_GATE:-true}"
FEDCD_ENTROPY_OOD_SCALE="${FEDCD_ENTROPY_OOD_SCALE:-0.5}"
FEDCD_ENTROPY_OOD_USE_CLASS_STATS="${FEDCD_ENTROPY_OOD_USE_CLASS_STATS:-true}"
FEDCD_ENTROPY_OOD_CLASS_MIX="${FEDCD_ENTROPY_OOD_CLASS_MIX:-0.6}"
FEDCD_FUSION_MODE="${FEDCD_FUSION_MODE:-safe_logit}"
FEDCD_LEARNED_BLEND_ALPHA="${FEDCD_LEARNED_BLEND_ALPHA:-0.5}"
FEDCD_LEARNED_BLEND_AUTO_TUNE="${FEDCD_LEARNED_BLEND_AUTO_TUNE:-false}"
FEDCD_LEARNED_BLEND_PERIOD="${FEDCD_LEARNED_BLEND_PERIOD:-1}"
FEDCD_LEARNED_BLEND_CANDIDATES="${FEDCD_LEARNED_BLEND_CANDIDATES:-11}"
FEDCD_SAFE_FUSION_HIDDEN_DIM="${FEDCD_SAFE_FUSION_HIDDEN_DIM:-64}"
FEDCD_SAFE_FUSION_DROPOUT="${FEDCD_SAFE_FUSION_DROPOUT:-0.0}"
FEDCD_SAFE_FUSION_PM_PRIOR="${FEDCD_SAFE_FUSION_PM_PRIOR:-0.8}"
FEDCD_SAFE_FUSION_TEMPERATURE="${FEDCD_SAFE_FUSION_TEMPERATURE:-1.0}"
FEDCD_SAFE_FUSION_MIN_PM_WEIGHT="${FEDCD_SAFE_FUSION_MIN_PM_WEIGHT:-0.0}"
FEDCD_SAFE_FUSION_MAX_PM_WEIGHT="${FEDCD_SAFE_FUSION_MAX_PM_WEIGHT:-1.0}"
FEDCD_SAFE_FUSION_LOSS_WEIGHT="${FEDCD_SAFE_FUSION_LOSS_WEIGHT:-0.3}"
FEDCD_SAFE_FUSION_MARGIN="${FEDCD_SAFE_FUSION_MARGIN:-0.02}"
FEDCD_SAFE_FUSION_LR_SCALE="${FEDCD_SAFE_FUSION_LR_SCALE:-1.0}"
FEDCD_SAFE_FUSION_ROUTE_WEIGHT="${FEDCD_SAFE_FUSION_ROUTE_WEIGHT:-0.6}"
FEDCD_SAFE_FUSION_ROUTE_TAU="${FEDCD_SAFE_FUSION_ROUTE_TAU:-0.2}"
FEDCD_SAFE_FUSION_ROUTE_FLOOR="${FEDCD_SAFE_FUSION_ROUTE_FLOOR:-0.05}"
FEDCD_SAFE_FUSION_ROUTE_GAP_POWER="${FEDCD_SAFE_FUSION_ROUTE_GAP_POWER:-1.0}"
FEDCD_PM_DEFER_CONF_THRESHOLD="${FEDCD_PM_DEFER_CONF_THRESHOLD:-0.55}"
FEDCD_PM_DEFER_GM_MARGIN="${FEDCD_PM_DEFER_GM_MARGIN:-0.02}"
FEDCD_PM_DEFER_OOD_THRESHOLD="${FEDCD_PM_DEFER_OOD_THRESHOLD:-0.35}"
FEDCD_BRANCH_TEMP_CALIBRATION_ENABLE="${FEDCD_BRANCH_TEMP_CALIBRATION_ENABLE:-false}"
FEDCD_BRANCH_TEMP_CALIBRATION_PERIOD="${FEDCD_BRANCH_TEMP_CALIBRATION_PERIOD:-1}"
FEDCD_BRANCH_TEMP_CALIBRATION_STEPS="${FEDCD_BRANCH_TEMP_CALIBRATION_STEPS:-80}"
FEDCD_BRANCH_TEMP_CALIBRATION_LR="${FEDCD_BRANCH_TEMP_CALIBRATION_LR:-0.05}"
FEDCD_BRANCH_TEMP_MIN="${FEDCD_BRANCH_TEMP_MIN:-0.5}"
FEDCD_BRANCH_TEMP_MAX="${FEDCD_BRANCH_TEMP_MAX:-5.0}"
FEDCD_BRANCH_TEMP_SAMPLES="${FEDCD_BRANCH_TEMP_SAMPLES:-512}"
FEDCD_ROUTER_ENABLE="${FEDCD_ROUTER_ENABLE:-false}"
FEDCD_ROUTER_TYPE="${FEDCD_ROUTER_TYPE:-attention}"
FEDCD_ROUTER_HIDDEN_DIM="${FEDCD_ROUTER_HIDDEN_DIM:-128}"
FEDCD_ROUTER_ATTN_DIM="${FEDCD_ROUTER_ATTN_DIM:-128}"
FEDCD_ROUTER_ATTN_HEADS="${FEDCD_ROUTER_ATTN_HEADS:-4}"
FEDCD_ROUTER_DROPOUT="${FEDCD_ROUTER_DROPOUT:-0.0}"
FEDCD_ROUTER_LR_SCALE="${FEDCD_ROUTER_LR_SCALE:-1.0}"
FEDCD_ROUTER_LOSS_WEIGHT="${FEDCD_ROUTER_LOSS_WEIGHT:-0.0}"
FEDCD_ROUTER_THRESHOLD="${FEDCD_ROUTER_THRESHOLD:-0.55}"
FEDCD_ROUTER_TEMPERATURE="${FEDCD_ROUTER_TEMPERATURE:-1.0}"
FEDCD_ROUTER_NEG_STD_SCALE="${FEDCD_ROUTER_NEG_STD_SCALE:-2.0}"
FEDCD_ROUTER_USE_FEATURE_NORM="${FEDCD_ROUTER_USE_FEATURE_NORM:-true}"
FEDCD_ROUTER_MIN_GM_WEIGHT="${FEDCD_ROUTER_MIN_GM_WEIGHT:-0.10}"
FEDCD_ROUTER_CONF_DEFER_MARGIN="${FEDCD_ROUTER_CONF_DEFER_MARGIN:-0.03}"
FEDCD_ROUTER_CONF_DEFER_STRENGTH="${FEDCD_ROUTER_CONF_DEFER_STRENGTH:-0.7}"
FEDCD_ROUTER_CONF_DEFER_TAU="${FEDCD_ROUTER_CONF_DEFER_TAU:-0.1}"
FEDCD_ROUTER_REINIT_ON_INITIAL_BROADCAST="${FEDCD_ROUTER_REINIT_ON_INITIAL_BROADCAST:-true}"
FEDCD_ROUTER_SUPERVISION_MODE="${FEDCD_ROUTER_SUPERVISION_MODE:-hybrid}"
FEDCD_ROUTER_BRANCH_MARGIN="${FEDCD_ROUTER_BRANCH_MARGIN:-0.02}"
FEDCD_ROUTER_GAP_POWER="${FEDCD_ROUTER_GAP_POWER:-1.0}"
FEDCD_ROUTER_MIN_LABELED_SAMPLES="${FEDCD_ROUTER_MIN_LABELED_SAMPLES:-4}"
FEDCD_ROUTER_FALLBACK_OOD_LOSS="${FEDCD_ROUTER_FALLBACK_OOD_LOSS:-true}"
FEDCD_ROUTER_USE_VAL_SPLIT="${FEDCD_ROUTER_USE_VAL_SPLIT:-true}"
FEDCD_ROUTER_VAL_RATIO="${FEDCD_ROUTER_VAL_RATIO:-0.1}"
FEDCD_ROUTER_VAL_MIN_SAMPLES="${FEDCD_ROUTER_VAL_MIN_SAMPLES:-32}"
FEDCD_ROUTER_VAL_MAX_SAMPLES="${FEDCD_ROUTER_VAL_MAX_SAMPLES:-512}"
FEDCD_ROUTER_SOFT_TAU="${FEDCD_ROUTER_SOFT_TAU:-0.2}"
FEDCD_ROUTER_SOFT_LABEL_FLOOR="${FEDCD_ROUTER_SOFT_LABEL_FLOOR:-0.05}"
FEDCD_ROUTER_BALANCE_WEIGHT="${FEDCD_ROUTER_BALANCE_WEIGHT:-0.1}"
FEDCD_ROUTER_BALANCE_TARGET="${FEDCD_ROUTER_BALANCE_TARGET:-0.55}"
FEDCD_ROUTER_BALANCE_TOLERANCE="${FEDCD_ROUTER_BALANCE_TOLERANCE:-0.2}"
FEDCD_ROUTER_SERVER_DISTILL_ENABLE="${FEDCD_ROUTER_SERVER_DISTILL_ENABLE:-false}"
FEDCD_ROUTER_SERVER_NEG_MODE="${FEDCD_ROUTER_SERVER_NEG_MODE:-all_other}"
FEDCD_ROUTER_SERVER_NEG_TOPK="${FEDCD_ROUTER_SERVER_NEG_TOPK:-2}"
FEDCD_ROUTER_SERVER_SYNTH_WEIGHT="${FEDCD_ROUTER_SERVER_SYNTH_WEIGHT:-0.0}"
FEDCD_ROUTER_SERVER_SYNTH_SAMPLES="${FEDCD_ROUTER_SERVER_SYNTH_SAMPLES:-128}"
FEDCD_LOCAL_PM_ONLY_OBJECTIVE="${FEDCD_LOCAL_PM_ONLY_OBJECTIVE:-false}"
FEDCD_GM_UPDATE_MODE="${FEDCD_GM_UPDATE_MODE:-server_pm_teacher}"
FEDCD_PM_TO_GM_MASK_ENABLE="${FEDCD_PM_TO_GM_MASK_ENABLE:-false}"
FEDCD_PM_TO_GM_MASK_UNIFIED="${FEDCD_PM_TO_GM_MASK_UNIFIED:-true}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "FedCD baseline-aligned run (GM224/PM1322/SmallFExt + routing-focused fusion)"
echo "dataset=${DATASET} nc=${NUM_CLIENTS} gr=${GLOBAL_ROUNDS} ls=${LOCAL_EPOCHS} lr=${LR} lbs=${BATCH_SIZE}"
echo "model: GM=${GM_MODEL}, PM=${PM_MODEL}, FExt=${FEXT_MODEL}"
echo "teacher: source=${TEACHER_SOURCE}, topk=${TEACHER_TOPK}, min_active=${TEACHER_MIN_ACTIVE}"
echo "=========================================================="

cd "${ROOT_DIR}"
CONDA_NO_PLUGINS=true conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u system/main.py \
    -go "${GOAL}" \
    -algo FedCD \
    -dev cuda -did "${GPU_ID}" \
    -data "${DATASET}" \
    -ncl "${NUM_CLASSES}" \
    -lbs "${BATCH_SIZE}" \
    --num_workers 0 \
    --pin_memory true \
    --prefetch_factor 2 \
    --gpu_batch_mult 1 \
    --gpu_batch_max 0 \
    --amp true \
    --tf32 true \
    --log_usage true \
    --log_usage_every 1 \
    -lr "${LR}" \
    -ld false \
    -ldg 0.99 \
    -gr "${GLOBAL_ROUNDS}" \
    -tc 100 \
    -ls "${LOCAL_EPOCHS}" \
    -jr "${JOIN_RATIO}" \
    -rjr false \
    -nc "${NUM_CLIENTS}" \
    --num_clusters 5 \
    --cluster_threshold 0.1 \
    --adaptive_threshold true \
    --threshold_step 0.01 \
    --threshold_step_max 0.1 \
    --threshold_decay 0.9 \
    --act_window_size 5 \
    --act_min_slope 0.0002 \
    --threshold_inc_rate 1.3 \
    --threshold_dec_rate 0.5 \
    --threshold_max 0.95 \
    --ema_alpha 0.3 \
    --tolerance_ratio 0.4 \
    --cluster_period 2 \
    --pm_period 1 \
    --global_period 2 \
    --fedcd_enable_clustering true \
    --cluster_sample_size 512 \
    --max_dynamic_clusters 0 \
    --fedcd_nc_weight 0.02 \
    --fedcd_nc_target_corr -0.1 \
    --fedcd_fusion_weight 0.6 \
    --fedcd_pm_logits_weight 0.7 \
    --fedcd_pm_only_weight 1.5 \
    --fedcd_gm_logits_weight 0.0 \
    --fedcd_local_pm_only_objective "${FEDCD_LOCAL_PM_ONLY_OBJECTIVE}" \
    --fedcd_gm_lr_scale 0.1 \
    --fedcd_gm_update_mode "${FEDCD_GM_UPDATE_MODE}" \
    --fedcd_pm_to_gm_mask_enable "${FEDCD_PM_TO_GM_MASK_ENABLE}" \
    --fedcd_pm_to_gm_mask_unified "${FEDCD_PM_TO_GM_MASK_UNIFIED}" \
    --fedcd_hybrid_proto_blend 0.35 \
    --fedcd_entropy_temp_pm "${FEDCD_ENTROPY_TEMP_PM}" \
    --fedcd_entropy_temp_gm "${FEDCD_ENTROPY_TEMP_GM}" \
    --fedcd_entropy_min_pm_weight "${FEDCD_ENTROPY_MIN_PM_WEIGHT}" \
    --fedcd_entropy_max_pm_weight "${FEDCD_ENTROPY_MAX_PM_WEIGHT}" \
    --fedcd_entropy_gate_tau "${FEDCD_ENTROPY_GATE_TAU}" \
    --fedcd_entropy_pm_bias "${FEDCD_ENTROPY_PM_BIAS}" \
    --fedcd_entropy_gm_bias "${FEDCD_ENTROPY_GM_BIAS}" \
    --fedcd_entropy_disagree_gm_boost "${FEDCD_ENTROPY_DISAGREE_GM_BOOST}" \
    --fedcd_entropy_use_class_reliability "${FEDCD_ENTROPY_USE_CLASS_RELIABILITY}" \
    --fedcd_entropy_reliability_scale "${FEDCD_ENTROPY_RELIABILITY_SCALE}" \
    --fedcd_entropy_hard_switch_margin "${FEDCD_ENTROPY_HARD_SWITCH_MARGIN}" \
    --fedcd_entropy_use_ood_gate "${FEDCD_ENTROPY_USE_OOD_GATE}" \
    --fedcd_entropy_ood_scale "${FEDCD_ENTROPY_OOD_SCALE}" \
    --fedcd_entropy_ood_use_class_stats "${FEDCD_ENTROPY_OOD_USE_CLASS_STATS}" \
    --fedcd_entropy_ood_class_mix "${FEDCD_ENTROPY_OOD_CLASS_MIX}" \
    --fedcd_fusion_mode "${FEDCD_FUSION_MODE}" \
    --fedcd_learned_blend_alpha "${FEDCD_LEARNED_BLEND_ALPHA}" \
    --fedcd_learned_blend_auto_tune "${FEDCD_LEARNED_BLEND_AUTO_TUNE}" \
    --fedcd_learned_blend_period "${FEDCD_LEARNED_BLEND_PERIOD}" \
    --fedcd_learned_blend_candidates "${FEDCD_LEARNED_BLEND_CANDIDATES}" \
    --fedcd_safe_fusion_hidden_dim "${FEDCD_SAFE_FUSION_HIDDEN_DIM}" \
    --fedcd_safe_fusion_dropout "${FEDCD_SAFE_FUSION_DROPOUT}" \
    --fedcd_safe_fusion_pm_prior "${FEDCD_SAFE_FUSION_PM_PRIOR}" \
    --fedcd_safe_fusion_temperature "${FEDCD_SAFE_FUSION_TEMPERATURE}" \
    --fedcd_safe_fusion_min_pm_weight "${FEDCD_SAFE_FUSION_MIN_PM_WEIGHT}" \
    --fedcd_safe_fusion_max_pm_weight "${FEDCD_SAFE_FUSION_MAX_PM_WEIGHT}" \
    --fedcd_safe_fusion_loss_weight "${FEDCD_SAFE_FUSION_LOSS_WEIGHT}" \
    --fedcd_safe_fusion_margin "${FEDCD_SAFE_FUSION_MARGIN}" \
    --fedcd_safe_fusion_lr_scale "${FEDCD_SAFE_FUSION_LR_SCALE}" \
    --fedcd_safe_fusion_route_weight "${FEDCD_SAFE_FUSION_ROUTE_WEIGHT}" \
    --fedcd_safe_fusion_route_tau "${FEDCD_SAFE_FUSION_ROUTE_TAU}" \
    --fedcd_safe_fusion_route_floor "${FEDCD_SAFE_FUSION_ROUTE_FLOOR}" \
    --fedcd_safe_fusion_route_gap_power "${FEDCD_SAFE_FUSION_ROUTE_GAP_POWER}" \
    --fedcd_pm_defer_conf_threshold "${FEDCD_PM_DEFER_CONF_THRESHOLD}" \
    --fedcd_pm_defer_gm_margin "${FEDCD_PM_DEFER_GM_MARGIN}" \
    --fedcd_pm_defer_ood_threshold "${FEDCD_PM_DEFER_OOD_THRESHOLD}" \
    --fedcd_branch_temp_calibration_enable "${FEDCD_BRANCH_TEMP_CALIBRATION_ENABLE}" \
    --fedcd_branch_temp_calibration_period "${FEDCD_BRANCH_TEMP_CALIBRATION_PERIOD}" \
    --fedcd_branch_temp_calibration_steps "${FEDCD_BRANCH_TEMP_CALIBRATION_STEPS}" \
    --fedcd_branch_temp_calibration_lr "${FEDCD_BRANCH_TEMP_CALIBRATION_LR}" \
    --fedcd_branch_temp_min "${FEDCD_BRANCH_TEMP_MIN}" \
    --fedcd_branch_temp_max "${FEDCD_BRANCH_TEMP_MAX}" \
    --fedcd_branch_temp_samples "${FEDCD_BRANCH_TEMP_SAMPLES}" \
    --fedcd_router_enable "${FEDCD_ROUTER_ENABLE}" \
    --fedcd_router_type "${FEDCD_ROUTER_TYPE}" \
    --fedcd_router_hidden_dim "${FEDCD_ROUTER_HIDDEN_DIM}" \
    --fedcd_router_attn_dim "${FEDCD_ROUTER_ATTN_DIM}" \
    --fedcd_router_attn_heads "${FEDCD_ROUTER_ATTN_HEADS}" \
    --fedcd_router_dropout "${FEDCD_ROUTER_DROPOUT}" \
    --fedcd_router_lr_scale "${FEDCD_ROUTER_LR_SCALE}" \
    --fedcd_router_loss_weight "${FEDCD_ROUTER_LOSS_WEIGHT}" \
    --fedcd_router_threshold "${FEDCD_ROUTER_THRESHOLD}" \
    --fedcd_router_temperature "${FEDCD_ROUTER_TEMPERATURE}" \
    --fedcd_router_neg_std_scale "${FEDCD_ROUTER_NEG_STD_SCALE}" \
    --fedcd_router_use_feature_norm "${FEDCD_ROUTER_USE_FEATURE_NORM}" \
    --fedcd_router_min_gm_weight "${FEDCD_ROUTER_MIN_GM_WEIGHT}" \
    --fedcd_router_conf_defer_margin "${FEDCD_ROUTER_CONF_DEFER_MARGIN}" \
    --fedcd_router_conf_defer_strength "${FEDCD_ROUTER_CONF_DEFER_STRENGTH}" \
    --fedcd_router_conf_defer_tau "${FEDCD_ROUTER_CONF_DEFER_TAU}" \
    --fedcd_router_reinit_on_initial_broadcast "${FEDCD_ROUTER_REINIT_ON_INITIAL_BROADCAST}" \
    --fedcd_router_supervision_mode "${FEDCD_ROUTER_SUPERVISION_MODE}" \
    --fedcd_router_branch_margin "${FEDCD_ROUTER_BRANCH_MARGIN}" \
    --fedcd_router_gap_power "${FEDCD_ROUTER_GAP_POWER}" \
    --fedcd_router_min_labeled_samples "${FEDCD_ROUTER_MIN_LABELED_SAMPLES}" \
    --fedcd_router_fallback_ood_loss "${FEDCD_ROUTER_FALLBACK_OOD_LOSS}" \
    --fedcd_router_use_val_split "${FEDCD_ROUTER_USE_VAL_SPLIT}" \
    --fedcd_router_val_ratio "${FEDCD_ROUTER_VAL_RATIO}" \
    --fedcd_router_val_min_samples "${FEDCD_ROUTER_VAL_MIN_SAMPLES}" \
    --fedcd_router_val_max_samples "${FEDCD_ROUTER_VAL_MAX_SAMPLES}" \
    --fedcd_router_soft_tau "${FEDCD_ROUTER_SOFT_TAU}" \
    --fedcd_router_soft_label_floor "${FEDCD_ROUTER_SOFT_LABEL_FLOOR}" \
    --fedcd_router_balance_weight "${FEDCD_ROUTER_BALANCE_WEIGHT}" \
    --fedcd_router_balance_target "${FEDCD_ROUTER_BALANCE_TARGET}" \
    --fedcd_router_balance_tolerance "${FEDCD_ROUTER_BALANCE_TOLERANCE}" \
    --fedcd_router_server_distill_enable "${FEDCD_ROUTER_SERVER_DISTILL_ENABLE}" \
    --fedcd_router_server_neg_mode "${FEDCD_ROUTER_SERVER_NEG_MODE}" \
    --fedcd_router_server_neg_topk "${FEDCD_ROUTER_SERVER_NEG_TOPK}" \
    --fedcd_router_server_synth_weight "${FEDCD_ROUTER_SERVER_SYNTH_WEIGHT}" \
    --fedcd_router_server_synth_samples "${FEDCD_ROUTER_SERVER_SYNTH_SAMPLES}" \
    --fedcd_gate_reliability_ema 0.9 \
    --fedcd_gate_reliability_samples 512 \
    --fedcd_gate_feature_ema 0.9 \
    --fedcd_gate_feature_samples 512 \
    --fedcd_warmup_epochs 0 \
    --fedcd_pm_teacher_lr 0.008 \
    --fedcd_pm_teacher_temp 2.0 \
    --fedcd_pm_teacher_kl_weight 1.0 \
    --fedcd_pm_teacher_ce_weight 0.0 \
    --fedcd_pm_teacher_samples "${FEDCD_PM_TEACHER_SAMPLES}" \
    --fedcd_pm_teacher_batch_size "${FEDCD_PM_TEACHER_BATCH_SIZE}" \
    --fedcd_pm_teacher_epochs "${FEDCD_PM_TEACHER_EPOCHS}" \
    --fedcd_pm_teacher_proxy_dataset "${TEACHER_PROXY_DATASET}" \
    --fedcd_pm_teacher_proxy_root "" \
    --fedcd_pm_teacher_proxy_split train \
    --fedcd_pm_teacher_proxy_download false \
    --fedcd_pm_teacher_allow_test_fallback false \
    --fedcd_pm_teacher_source "${TEACHER_SOURCE}" \
    --fedcd_pm_teacher_confidence_weight true \
    --fedcd_pm_teacher_confidence_min 0.1 \
    --fedcd_pm_teacher_confidence_power 2.0 \
    --fedcd_pm_teacher_ensemble_confidence true \
    --fedcd_pm_teacher_topk "${TEACHER_TOPK}" \
    --fedcd_pm_teacher_abstain_threshold "${TEACHER_ABSTAIN_THRESHOLD}" \
    --fedcd_pm_teacher_teacher_abstain_threshold "${TEACHER_TEACHER_ABSTAIN_THRESHOLD}" \
    --fedcd_pm_teacher_min_active_teachers "${TEACHER_MIN_ACTIVE}" \
    --fedcd_pm_teacher_consensus_min_ratio "${TEACHER_CONSENSUS_MIN_RATIO}" \
    --fedcd_pm_teacher_correct_only false \
    --fedcd_pm_teacher_rel_weight 0.1 \
    --fedcd_pm_teacher_rel_batch 64 \
    --fedcd_init_pretrain true \
    --fedcd_init_epochs "${FEDCD_INIT_EPOCHS}" \
    --fedcd_init_lr 0.005 \
    --fedcd_init_samples "${FEDCD_INIT_SAMPLES}" \
    --fedcd_init_batch_size "${FEDCD_INIT_BATCH_SIZE}" \
    --fedcd_init_ce_weight 1.0 \
    --fedcd_init_kd_weight 1.0 \
    --fedcd_init_entropy_weight 0.05 \
    --fedcd_init_diversity_weight 0.05 \
    --fedcd_proto_teacher_lr 0.01 \
    --fedcd_proto_teacher_steps 200 \
    --fedcd_proto_teacher_batch_size 256 \
    --fedcd_proto_teacher_temp 2.0 \
    --fedcd_proto_teacher_ce_weight 1.0 \
    --fedcd_proto_teacher_kl_weight 0.5 \
    --fedcd_proto_teacher_noise_scale 1.0 \
    --fedcd_proto_teacher_min_count 1.0 \
    --fedcd_proto_teacher_client_samples 0 \
    --fedcd_proto_teacher_confidence_weight true \
    --fedcd_proto_teacher_confidence_min 0.05 \
    --fedcd_proto_teacher_confidence_power 1.0 \
    --fedcd_search_enable false \
    --fedcd_search_min_rounds 8 \
    --fedcd_search_patience 6 \
    --fedcd_search_drop_patience 3 \
    --fedcd_search_drop_delta 0.003 \
    --fedcd_search_score_gm_weight 0.75 \
    --fedcd_search_score_pm_weight 0.25 \
    --fedcd_search_score_eps 0.0001 \
    --fedcd_search_min_pm_local_acc 0.55 \
    --fedcd_search_min_gm_global_acc 0.18 \
    --gm_model "${GM_MODEL}" \
    --pm_model "${PM_MODEL}" \
    --fext_model "${FEXT_MODEL}" \
    --fext_dim "${FEXT_DIM}" \
    --eval_common_global true \
    --global_test_samples 0 \
    --common_eval_batch_size 256 \
    -pv 0 \
    -t 1 \
    -eg 1 \
    -sfn items \
    -ab false \
    -dlg false \
    -dlgg 100 \
    -bnpc 2 \
    -nnc 0 \
    -ften 0 \
    -fd 512 \
    -vs 80 \
    -ml 200 \
    -fs 0 \
    -oom true \
    -cdr 0.0 \
    -tsr 0.0 \
    -ssr 0.0 \
    -ts false \
    -tth 10000
