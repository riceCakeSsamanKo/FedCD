#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_comm_compare_fedcd_vgg8w768_vs_baseline.sh
#   CONDA_ENV=pfllib GPU_ID=1 BASELINE_ALGOS="FedAS Local FedAvg FedProx FedKD" bash run_comm_compare_fedcd_vgg8w768_vs_baseline.sh

CONDA_ENV="${CONDA_ENV:-pfllib}"
GPU_ID="${GPU_ID:-0}"
BASELINE_ALGOS="${BASELINE_ALGOS:-FedAS Local FedAvg FedProx FedKD}"
BASELINE_MODEL="${BASELINE_MODEL:-VGG8}"
FEDCD_GM_MODEL="${FEDCD_GM_MODEL:-VGG8W512}"
FEDCD_PM_MODEL="${FEDCD_PM_MODEL:-VGG8W1024}"
FEDCD_TEACHER_PROXY_DATASET="${FEDCD_TEACHER_PROXY_DATASET:-Cifar100}"
FEDCD_TEACHER_SOURCE="${FEDCD_TEACHER_SOURCE:-cluster}"
FEDCD_TEACHER_CORRECT_ONLY="${FEDCD_TEACHER_CORRECT_ONLY:-false}"
FEDCD_TEACHER_CONF_MIN="${FEDCD_TEACHER_CONF_MIN:-0.1}"
FEDCD_TEACHER_CONF_POWER="${FEDCD_TEACHER_CONF_POWER:-2.0}"
# Use all cluster representative PMs as teachers by default.
FEDCD_TEACHER_TOPK="${FEDCD_TEACHER_TOPK:-0}"
FEDCD_TEACHER_ABSTAIN_THRESHOLD="${FEDCD_TEACHER_ABSTAIN_THRESHOLD:-0.0}"
FEDCD_TEACHER_TEACHER_ABSTAIN_THRESHOLD="${FEDCD_TEACHER_TEACHER_ABSTAIN_THRESHOLD:-0.0}"
FEDCD_TEACHER_MIN_ACTIVE="${FEDCD_TEACHER_MIN_ACTIVE:-1}"
FEDCD_TEACHER_CONSENSUS_MIN_RATIO="${FEDCD_TEACHER_CONSENSUS_MIN_RATIO:-0.0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
COMPARE_ROOT="${COMPARE_ROOT:-${ROOT_DIR}/compare}"
COMPARE_RUN_DIR="${COMPARE_ROOT}/${RUN_TS}"
LOG_DIR="${LOG_DIR:-${COMPARE_RUN_DIR}/console}"
mkdir -p "${LOG_DIR}"
mkdir -p "${COMPARE_RUN_DIR}"

run_py() {
  CONDA_NO_PLUGINS=true conda run --no-capture-output -n "${CONDA_ENV}" python -u "$@"
}

run_py_in_dir() {
  local workdir="$1"
  shift
  (
    cd "${workdir}"
    CONDA_NO_PLUGINS=true conda run --no-capture-output -n "${CONDA_ENV}" python -u "$@"
  )
}

latest_named_file_after() {
  local search_root="$1"
  local stamp_file="$2"
  local file_name="$3"
  find "${search_root}" -type f -name "${file_name}" -newer "${stamp_file}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | awk 'END{print $2}'
}

latest_named_file() {
  local search_root="$1"
  local file_name="$2"
  find "${search_root}" -type f -name "${file_name}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | awk 'END{print $2}'
}

collect_csv_for_method() {
  local method="$1"
  local search_root="$2"
  local stamp_file="$3"
  local method_dir="${COMPARE_RUN_DIR}/${method}"
  mkdir -p "${method_dir}"

  local acc_path
  acc_path="$(latest_named_file_after "${search_root}" "${stamp_file}" "acc.csv")"
  if [ -z "${acc_path}" ]; then
    acc_path="$(latest_named_file "${search_root}" "acc.csv")"
  fi

  if [ -n "${acc_path}" ]; then
    cp "${acc_path}" "${method_dir}/acc.csv"
    printf '%s\n' "${acc_path}" > "${method_dir}/source_acc_path.txt"
  else
    echo "[Warn] ${method}: acc.csv not found under ${search_root}"
  fi
}

collect_optional_csv() {
  local method="$1"
  local search_root="$2"
  local stamp_file="$3"
  local file_name="$4"
  local method_dir="${COMPARE_RUN_DIR}/${method}"
  mkdir -p "${method_dir}"

  local src
  src="$(latest_named_file_after "${search_root}" "${stamp_file}" "${file_name}")"
  if [ -z "${src}" ]; then
    src="$(latest_named_file "${search_root}" "${file_name}")"
  fi
  if [ -n "${src}" ]; then
    cp "${src}" "${method_dir}/${file_name}"
  fi
}

BASELINE_ALGOS="${BASELINE_ALGOS//,/ }"
read -r -a BASELINE_ALGO_LIST <<< "${BASELINE_ALGOS}"
if [ "${#BASELINE_ALGO_LIST[@]}" -eq 0 ]; then
  echo "No baseline algorithms provided. Set BASELINE_ALGOS."
  exit 1
fi

echo "[1/2] Baseline communication runs: ${BASELINE_ALGO_LIST[*]}"
for algo in "${BASELINE_ALGO_LIST[@]}"; do
  echo "[Baseline] Running ${algo}"
  stamp_file="$(mktemp)"
  touch "${stamp_file}"
  run_py_in_dir "${ROOT_DIR}/FedCD-Baseline" system/main.py \
    -go train \
    -algo "${algo}" \
    -dev cuda -did "${GPU_ID}" \
    -data Cifar10_pat_nc20 \
    -ncl 10 \
    -m "${BASELINE_MODEL}" \
    -lbs 128 \
    -lr 0.005 \
    -gr 5 \
    -ls 2 \
    -jr 1.0 \
    -nc 20 \
    -eg 1 \
    -t 1 2>&1 | tee "${LOG_DIR}/${RUN_TS}_baseline_${algo}.log"
  collect_csv_for_method "${algo}" "${ROOT_DIR}/FedCD-Baseline/logs/${algo}" "${stamp_file}"
  rm -f "${stamp_file}"
done

echo "[2/2] FedCD communication run (194429 settings + GM=${FEDCD_GM_MODEL}, PM=${FEDCD_PM_MODEL}, teacher_source=${FEDCD_TEACHER_SOURCE}, ls=2, gr=5)"
stamp_file="$(mktemp)"
touch "${stamp_file}"
run_py_in_dir "${ROOT_DIR}/FedCD-Core" system/main.py \
  -go train \
  -algo FedCD \
  -dev cuda -did "${GPU_ID}" \
  -data Cifar10_pat_nc20 \
  -ncl 10 \
  -lbs 128 \
  --num_workers 0 \
  --pin_memory true \
  --prefetch_factor 2 \
  --gpu_batch_mult 1 \
  --gpu_batch_max 0 \
  --amp true \
  --tf32 true \
  --log_usage true \
  --log_usage_every 1 \
  -lr 0.005 \
  -ld false \
  -ldg 0.99 \
  -gr 5 \
  -tc 100 \
  -ls 2 \
  -jr 1.0 \
  -rjr false \
  -nc 20 \
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
  --fedcd_local_pm_only_objective true \
  --fedcd_gm_lr_scale 0.1 \
  --fedcd_gm_update_mode server_pm_teacher \
  --fedcd_hybrid_proto_blend 0.35 \
  --fedcd_entropy_temp_pm 1.0 \
  --fedcd_entropy_temp_gm 1.0 \
  --fedcd_entropy_min_pm_weight 0.5 \
  --fedcd_entropy_max_pm_weight 0.5 \
  --fedcd_entropy_gate_tau 0.15 \
  --fedcd_entropy_pm_bias 0.0 \
  --fedcd_entropy_gm_bias 0.0 \
  --fedcd_entropy_disagree_gm_boost 0.0 \
  --fedcd_entropy_use_class_reliability false \
  --fedcd_entropy_reliability_scale 0.8 \
  --fedcd_entropy_hard_switch_margin 0.35 \
  --fedcd_entropy_use_ood_gate true \
  --fedcd_entropy_ood_scale 1.0 \
  --fedcd_gate_reliability_ema 0.9 \
  --fedcd_gate_reliability_samples 512 \
  --fedcd_gate_feature_ema 0.9 \
  --fedcd_gate_feature_samples 512 \
  --fedcd_warmup_epochs 0 \
  --fedcd_pm_teacher_lr 0.008 \
  --fedcd_pm_teacher_temp 2.0 \
  --fedcd_pm_teacher_kl_weight 1.0 \
  --fedcd_pm_teacher_ce_weight 0.0 \
  --fedcd_pm_teacher_samples 50000 \
  --fedcd_pm_teacher_batch_size 256 \
  --fedcd_pm_teacher_epochs 6 \
  --fedcd_pm_teacher_proxy_dataset "${FEDCD_TEACHER_PROXY_DATASET}" \
  --fedcd_pm_teacher_proxy_root "" \
  --fedcd_pm_teacher_proxy_split train \
  --fedcd_pm_teacher_proxy_download false \
  --fedcd_pm_teacher_allow_test_fallback false \
  --fedcd_pm_teacher_source "${FEDCD_TEACHER_SOURCE}" \
  --fedcd_pm_teacher_confidence_weight true \
  --fedcd_pm_teacher_confidence_min "${FEDCD_TEACHER_CONF_MIN}" \
  --fedcd_pm_teacher_confidence_power "${FEDCD_TEACHER_CONF_POWER}" \
  --fedcd_pm_teacher_ensemble_confidence true \
  --fedcd_pm_teacher_topk "${FEDCD_TEACHER_TOPK}" \
  --fedcd_pm_teacher_abstain_threshold "${FEDCD_TEACHER_ABSTAIN_THRESHOLD}" \
  --fedcd_pm_teacher_teacher_abstain_threshold "${FEDCD_TEACHER_TEACHER_ABSTAIN_THRESHOLD}" \
  --fedcd_pm_teacher_min_active_teachers "${FEDCD_TEACHER_MIN_ACTIVE}" \
  --fedcd_pm_teacher_consensus_min_ratio "${FEDCD_TEACHER_CONSENSUS_MIN_RATIO}" \
  --fedcd_pm_teacher_correct_only "${FEDCD_TEACHER_CORRECT_ONLY}" \
  --fedcd_pm_teacher_rel_weight 0.1 \
  --fedcd_pm_teacher_rel_batch 64 \
  --fedcd_init_pretrain true \
  --fedcd_init_epochs 5 \
  --fedcd_init_lr 0.005 \
  --fedcd_init_samples 50000 \
  --fedcd_init_batch_size 256 \
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
  --fedcd_search_enable true \
  --fedcd_search_min_rounds 8 \
  --fedcd_search_patience 6 \
  --fedcd_search_drop_patience 3 \
  --fedcd_search_drop_delta 0.003 \
  --fedcd_search_score_gm_weight 0.75 \
  --fedcd_search_score_pm_weight 0.25 \
  --fedcd_search_score_eps 0.0001 \
  --fedcd_search_min_pm_local_acc 0.55 \
  --fedcd_search_min_gm_global_acc 0.18 \
  --gm_model "${FEDCD_GM_MODEL}" \
  --pm_model "${FEDCD_PM_MODEL}" \
  --fext_model SmallFExt \
  --fext_dim 512 \
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
  -tth 10000 2>&1 | tee "${LOG_DIR}/${RUN_TS}_fedcd.log"
collect_csv_for_method "FedCD" "${ROOT_DIR}/FedCD-Core/logs/FedCD" "${stamp_file}"
collect_optional_csv "FedCD" "${ROOT_DIR}/FedCD-Core/logs/FedCD" "${stamp_file}" "cluster_acc.csv"
collect_optional_csv "FedCD" "${ROOT_DIR}/FedCD-Core/logs/FedCD" "${stamp_file}" "usage.csv"
rm -f "${stamp_file}"

echo "Done."
echo "Console logs:"
for algo in "${BASELINE_ALGO_LIST[@]}"; do
  echo "  ${LOG_DIR}/${RUN_TS}_baseline_${algo}.log"
done
echo "  ${LOG_DIR}/${RUN_TS}_fedcd.log"
echo "Collected CSV directory: ${COMPARE_RUN_DIR}"
for algo in "${BASELINE_ALGO_LIST[@]}"; do
  echo "  ${COMPARE_RUN_DIR}/${algo}/acc.csv"
done
echo "  ${COMPARE_RUN_DIR}/FedCD/acc.csv"
echo "Metrics: check each method acc.csv (and FedCD cluster_acc.csv/usage.csv if present)."
