#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="${ROOT_DIR}/FedCD-Core"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/compare/${STAMP}_routing_ablation"
CONSOLE_DIR="${OUT_DIR}/console"
mkdir -p "${CONSOLE_DIR}"

CONDA_ENV="${CONDA_ENV:-pfllib}"
GPU_ID="${GPU_ID:-0}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-20}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.005}"
DATASET="${DATASET:-Cifar10_pat_nc20}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"

GM_MODEL="${GM_MODEL:-VGG8W256}"
PM_MODEL="${PM_MODEL:-VGG8W1290}"
FEXT_MODEL="${FEXT_MODEL:-SmallFExt}"
FEXT_DIM="${FEXT_DIM:-256}"

SUMMARY_CSV="${OUT_DIR}/summary.csv"
cat > "${SUMMARY_CSV}" <<'CSV'
case,fusion_mode,router_enable,round,local_test_acc,pm_local_test_acc,gm_local_test_acc,global_test_acc,gm_only_global_test_acc,pm_global_test_acc,local_over_pm,global_over_gm,acc_path,console_log
CSV

latest_acc_path() {
  local model_dir="${CORE_DIR}/logs/FedCD/GM_${GM_MODEL}_PM_${PM_MODEL}_Fext_${FEXT_MODEL}/pat/NC_${NUM_CLIENTS}"
  ls -1t "${model_dir}"/date_*/time_*/acc.csv 2>/dev/null | head -n 1 || true
}

append_summary() {
  local case_name="$1"
  local fusion_mode="$2"
  local router_enable="$3"
  local log_path="$4"
  local acc_path
  local tail_line
  local ratio_local
  local ratio_global

  acc_path="$(latest_acc_path)"
  if [[ -z "${acc_path}" || ! -f "${acc_path}" ]]; then
    echo "${case_name},${fusion_mode},${router_enable},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${log_path}" >> "${SUMMARY_CSV}"
    return
  fi

  tail_line="$(tail -n 1 "${acc_path}")"
  ratio_local="$(python - <<'PY' "${tail_line}"
import sys
v=sys.argv[1].split(',')
local=float(v[1]); pml=float(v[2])
print(f"{(local/max(pml,1e-12)):.6f}")
PY
)"
  ratio_global="$(python - <<'PY' "${tail_line}"
import sys
v=sys.argv[1].split(',')
glob=float(v[4]); gmg=float(v[5])
print(f"{(glob/max(gmg,1e-12)):.6f}")
PY
)"

  echo "${case_name},${fusion_mode},${router_enable},${tail_line},${ratio_local},${ratio_global},${acc_path},${log_path}" >> "${SUMMARY_CSV}"
}

run_case() {
  local case_name="$1"
  shift
  local log_path="${CONSOLE_DIR}/${case_name}.log"
  echo ""
  echo "======================================================="
  echo "[${case_name}] start"
  echo "======================================================="
  (
    cd "${CORE_DIR}"
    env \
      CONDA_ENV="${CONDA_ENV}" \
      GPU_ID="${GPU_ID}" \
      DATASET="${DATASET}" \
      NUM_CLIENTS="${NUM_CLIENTS}" \
      GLOBAL_ROUNDS="${GLOBAL_ROUNDS}" \
      LOCAL_EPOCHS="${LOCAL_EPOCHS}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      LR="${LR}" \
      GM_MODEL="${GM_MODEL}" \
      PM_MODEL="${PM_MODEL}" \
      FEXT_MODEL="${FEXT_MODEL}" \
      FEXT_DIM="${FEXT_DIM}" \
      FEDCD_INIT_EPOCHS=2 \
      FEDCD_INIT_SAMPLES=10000 \
      FEDCD_PM_TEACHER_EPOCHS=3 \
      FEDCD_PM_TEACHER_SAMPLES=10000 \
      FEDCD_BRANCH_TEMP_CALIBRATION_ENABLE=false \
      "$@" \
      bash run.sh
  ) 2>&1 | tee "${log_path}"

  append_summary \
    "${case_name}" \
    "${FEDCD_FUSION_MODE:-unknown}" \
    "${FEDCD_ROUTER_ENABLE:-unknown}" \
    "${log_path}"
}

echo "Output dir: ${OUT_DIR}"
echo "Summary: ${SUMMARY_CSV}"
echo "Model: GM=${GM_MODEL}, PM=${PM_MODEL}, FExt=${FEXT_MODEL}"
echo "Rounds=${GLOBAL_ROUNDS}, LocalEpoch=${LOCAL_EPOCHS}, LR=${LR}, Batch=${BATCH_SIZE}"

# 1) Soft baseline (PM-heavy)
FEDCD_FUSION_MODE="soft"
FEDCD_ROUTER_ENABLE="false"
run_case "exp01_soft_pmheavy" \
  FEDCD_FUSION_MODE="${FEDCD_FUSION_MODE}" \
  FEDCD_ROUTER_ENABLE="${FEDCD_ROUTER_ENABLE}" \
  FEDCD_ENTROPY_MIN_PM_WEIGHT=0.80 \
  FEDCD_ENTROPY_MAX_PM_WEIGHT=0.98

# 2) Soft mid range
FEDCD_FUSION_MODE="soft"
FEDCD_ROUTER_ENABLE="false"
run_case "exp02_soft_mid" \
  FEDCD_FUSION_MODE="${FEDCD_FUSION_MODE}" \
  FEDCD_ROUTER_ENABLE="${FEDCD_ROUTER_ENABLE}" \
  FEDCD_ENTROPY_MIN_PM_WEIGHT=0.55 \
  FEDCD_ENTROPY_MAX_PM_WEIGHT=0.90

# 3) Soft balanced range
FEDCD_FUSION_MODE="soft"
FEDCD_ROUTER_ENABLE="false"
run_case "exp03_soft_balanced" \
  FEDCD_FUSION_MODE="${FEDCD_FUSION_MODE}" \
  FEDCD_ROUTER_ENABLE="${FEDCD_ROUTER_ENABLE}" \
  FEDCD_ENTROPY_MIN_PM_WEIGHT=0.35 \
  FEDCD_ENTROPY_MAX_PM_WEIGHT=0.75

# 4) Soft wide range
FEDCD_FUSION_MODE="soft"
FEDCD_ROUTER_ENABLE="false"
run_case "exp04_soft_wide" \
  FEDCD_FUSION_MODE="${FEDCD_FUSION_MODE}" \
  FEDCD_ROUTER_ENABLE="${FEDCD_ROUTER_ENABLE}" \
  FEDCD_ENTROPY_MIN_PM_WEIGHT=0.10 \
  FEDCD_ENTROPY_MAX_PM_WEIGHT=0.90

# 5) PM defer tuned
FEDCD_FUSION_MODE="pm_defer_hard"
FEDCD_ROUTER_ENABLE="false"
run_case "exp05_pm_defer_tuned" \
  FEDCD_FUSION_MODE="${FEDCD_FUSION_MODE}" \
  FEDCD_ROUTER_ENABLE="${FEDCD_ROUTER_ENABLE}" \
  FEDCD_PM_DEFER_CONF_THRESHOLD=0.45 \
  FEDCD_PM_DEFER_GM_MARGIN=0.04 \
  FEDCD_PM_DEFER_OOD_THRESHOLD=0.20

# 6) Router soft (trained) with guardrails
FEDCD_FUSION_MODE="router_soft"
FEDCD_ROUTER_ENABLE="true"
run_case "exp06_router_soft_trained" \
  FEDCD_FUSION_MODE="${FEDCD_FUSION_MODE}" \
  FEDCD_ROUTER_ENABLE="${FEDCD_ROUTER_ENABLE}" \
  FEDCD_ROUTER_LOSS_WEIGHT=0.20 \
  FEDCD_ROUTER_THRESHOLD=0.55 \
  FEDCD_ROUTER_MIN_GM_WEIGHT=0.20 \
  FEDCD_ROUTER_CONF_DEFER_MARGIN=0.05 \
  FEDCD_ROUTER_CONF_DEFER_STRENGTH=0.40 \
  FEDCD_ROUTER_CONF_DEFER_TAU=0.15 \
  FEDCD_ROUTER_SERVER_DISTILL_ENABLE=true \
  FEDCD_ROUTER_SERVER_NEG_MODE=farthest_k \
  FEDCD_ROUTER_SERVER_NEG_TOPK=2 \
  FEDCD_ROUTER_SERVER_SYNTH_WEIGHT=0.10 \
  FEDCD_ROUTER_SERVER_SYNTH_SAMPLES=128

echo ""
echo "Done. Summary at: ${SUMMARY_CSV}"
echo "Top candidates by balance:"
python - <<'PY' "${SUMMARY_CSV}"
import csv,sys
p=sys.argv[1]
rows=[]
with open(p) as f:
    r=csv.DictReader(f)
    for row in r:
        try:
            l=float(row["local_over_pm"])
            g=float(row["global_over_gm"])
            score=(l+g)/2
        except Exception:
            continue
        row["_score"]=score
        rows.append(row)
rows.sort(key=lambda x:x["_score"], reverse=True)
for row in rows[:3]:
    print(f"{row['case']}: score={row['_score']:.4f}, local/pm={row['local_over_pm']}, global/gm={row['global_over_gm']}")
PY
