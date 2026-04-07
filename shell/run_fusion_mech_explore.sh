#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="${ROOT_DIR}/FedCD-Core"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/compare/${TS}_fusion_mech_explore"
CONSOLE_DIR="${OUT_DIR}/console"
mkdir -p "${CONSOLE_DIR}"

# Base setting (override via env when needed)
CONDA_ENV="${CONDA_ENV:-pfllib}"
GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-Cifar10_pat_nc20}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-20}"
LR="${LR:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FEDCD_INIT_EPOCHS="${FEDCD_INIT_EPOCHS:-2}"
FEDCD_INIT_SAMPLES="${FEDCD_INIT_SAMPLES:-10000}"
FEDCD_INIT_BATCH_SIZE="${FEDCD_INIT_BATCH_SIZE:-256}"
FEDCD_PM_TEACHER_EPOCHS="${FEDCD_PM_TEACHER_EPOCHS:-3}"
FEDCD_PM_TEACHER_SAMPLES="${FEDCD_PM_TEACHER_SAMPLES:-10000}"
FEDCD_PM_TEACHER_BATCH_SIZE="${FEDCD_PM_TEACHER_BATCH_SIZE:-256}"

GM_MODEL="${GM_MODEL:-VGG8W256}"
PM_MODEL="${PM_MODEL:-VGG8W1290}"
FEXT_MODEL="${FEXT_MODEL:-SmallFExt}"
FEXT_DIM="${FEXT_DIM:-256}"

MODEL_LOG_DIR="${CORE_DIR}/logs/FedCD/GM_${GM_MODEL}_PM_${PM_MODEL}_Fext_${FEXT_MODEL}"
SUMMARY_CSV="${OUT_DIR}/summary.csv"

echo "case,fusion_mode,router_enable,temp_cal,learned_blend_auto,final_round,local_test_acc,pm_local_test_acc,gm_only_global_test_acc,global_test_acc,total_mb,acc_path,console_log" > "${SUMMARY_CSV}"

echo "Output dir: ${OUT_DIR}"
echo "Model dir : ${MODEL_LOG_DIR}"

latest_acc_path() {
  find "${MODEL_LOG_DIR}" -type f -name acc.csv -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

append_summary() {
  local case_name="$1"
  local fusion_mode="$2"
  local router_enable="$3"
  local temp_cal="$4"
  local learned_auto="$5"
  local acc_path="$6"
  local log_path="$7"

  if [[ -z "${acc_path}" || ! -f "${acc_path}" ]]; then
    echo "${case_name},${fusion_mode},${router_enable},${temp_cal},${learned_auto},NA,NA,NA,NA,NA,NA,NA,${log_path}" >> "${SUMMARY_CSV}"
    return
  fi

  local last
  last="$(tail -n 1 "${acc_path}")"
  local r local_acc pm_local gm_local global_acc gm_global pm_global train_loss uplink downlink total
  IFS=',' read -r r local_acc pm_local gm_local global_acc gm_global pm_global train_loss uplink downlink total <<< "${last}"

  echo "${case_name},${fusion_mode},${router_enable},${temp_cal},${learned_auto},${r},${local_acc},${pm_local},${gm_global},${global_acc},${total},${acc_path},${log_path}" >> "${SUMMARY_CSV}"
}

run_case() {
  local case_name="$1"
  local fusion_mode="$2"
  local router_enable="$3"
  local temp_cal="$4"
  local learned_auto="$5"
  shift 5
  local extra_env=("$@")

  local log_path="${CONSOLE_DIR}/${case_name}.log"
  echo "[RUN] ${case_name}"

  (
    cd "${CORE_DIR}"
    env \
      CONDA_ENV="${CONDA_ENV}" \
      GPU_ID="${GPU_ID}" \
      DATASET="${DATASET}" \
      NUM_CLIENTS="${NUM_CLIENTS}" \
      LOCAL_EPOCHS="${LOCAL_EPOCHS}" \
      GLOBAL_ROUNDS="${GLOBAL_ROUNDS}" \
      LR="${LR}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      FEDCD_INIT_EPOCHS="${FEDCD_INIT_EPOCHS}" \
      FEDCD_INIT_SAMPLES="${FEDCD_INIT_SAMPLES}" \
      FEDCD_INIT_BATCH_SIZE="${FEDCD_INIT_BATCH_SIZE}" \
      FEDCD_PM_TEACHER_EPOCHS="${FEDCD_PM_TEACHER_EPOCHS}" \
      FEDCD_PM_TEACHER_SAMPLES="${FEDCD_PM_TEACHER_SAMPLES}" \
      FEDCD_PM_TEACHER_BATCH_SIZE="${FEDCD_PM_TEACHER_BATCH_SIZE}" \
      GM_MODEL="${GM_MODEL}" \
      PM_MODEL="${PM_MODEL}" \
      FEXT_MODEL="${FEXT_MODEL}" \
      FEXT_DIM="${FEXT_DIM}" \
      FEDCD_FUSION_MODE="${fusion_mode}" \
      FEDCD_ROUTER_ENABLE="${router_enable}" \
      FEDCD_BRANCH_TEMP_CALIBRATION_ENABLE="${temp_cal}" \
      FEDCD_LEARNED_BLEND_AUTO_TUNE="${learned_auto}" \
      "${extra_env[@]}" \
      bash run.sh
  ) 2>&1 | tee "${log_path}"

  local acc_path
  acc_path="$(latest_acc_path)"
  append_summary "${case_name}" "${fusion_mode}" "${router_enable}" "${temp_cal}" "${learned_auto}" "${acc_path}" "${log_path}"
}

run_case \
  "exp01_soft_base" \
  "soft" \
  "false" \
  "false" \
  "false"

run_case \
  "exp02_soft_tempcal" \
  "soft" \
  "false" \
  "true" \
  "false" \
  FEDCD_BRANCH_TEMP_CALIBRATION_STEPS=60 \
  FEDCD_BRANCH_TEMP_SAMPLES=384

run_case \
  "exp03_poe_tempcal" \
  "poe_soft" \
  "false" \
  "true" \
  "false" \
  FEDCD_BRANCH_TEMP_CALIBRATION_STEPS=60 \
  FEDCD_BRANCH_TEMP_SAMPLES=384

run_case \
  "exp04_learned_blend" \
  "learned_blend" \
  "false" \
  "true" \
  "true" \
  FEDCD_LEARNED_BLEND_CANDIDATES=11 \
  FEDCD_BRANCH_TEMP_CALIBRATION_STEPS=60 \
  FEDCD_BRANCH_TEMP_SAMPLES=384

run_case \
  "exp05_pm_defer_tempcal" \
  "pm_defer_hard" \
  "false" \
  "true" \
  "false" \
  FEDCD_PM_DEFER_CONF_THRESHOLD=0.60 \
  FEDCD_PM_DEFER_GM_MARGIN=0.02 \
  FEDCD_PM_DEFER_OOD_THRESHOLD=0.35 \
  FEDCD_BRANCH_TEMP_CALIBRATION_STEPS=60 \
  FEDCD_BRANCH_TEMP_SAMPLES=384

run_case \
  "exp06_router_soft_hneg" \
  "router_soft" \
  "true" \
  "true" \
  "false" \
  FEDCD_ROUTER_TYPE=attention \
  FEDCD_ROUTER_LOSS_WEIGHT=0.2 \
  FEDCD_ROUTER_SERVER_DISTILL_ENABLE=true \
  FEDCD_ROUTER_SERVER_NEG_MODE=farthest_k \
  FEDCD_ROUTER_SERVER_NEG_TOPK=2 \
  FEDCD_ROUTER_SERVER_SYNTH_WEIGHT=0.3 \
  FEDCD_ROUTER_SERVER_SYNTH_SAMPLES=128 \
  FEDCD_BRANCH_TEMP_CALIBRATION_STEPS=40 \
  FEDCD_BRANCH_TEMP_SAMPLES=256

echo "Done. Summary: ${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
