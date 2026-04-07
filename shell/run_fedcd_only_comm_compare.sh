#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_fedcd_only_comm_compare.sh
#   CONDA_ENV=pfllib GPU_ID=0 bash run_fedcd_only_comm_compare.sh
#   MODEL_SPLITS="512:1024 640:896 768:768" TEACHER_MODES="cluster_all cluster_conf" bash run_fedcd_only_comm_compare.sh

CONDA_ENV="${CONDA_ENV:-pfllib}"
GPU_ID="${GPU_ID:-0}"

# Core training setup
DATASET="${DATASET:-Cifar10_pat_nc20}"
NUM_CLASSES="${NUM_CLASSES:-10}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
JOIN_RATIO="${JOIN_RATIO:-1.0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-20}"

# Keep PM+GM wrapper size near VGG8 by default (sum hidden = 1536)
# Format: "gm_hidden:pm_hidden ..."
MODEL_SPLITS="${MODEL_SPLITS:-256:1280 384:1152 512:1024 640:896 768:768}"

# Teacher mode candidates:
#   client_all   : all client PMs as teachers
#   cluster_all  : all cluster representative PMs as teachers
#   client_conf  : confident subset from client PMs
#   cluster_conf : confident subset from cluster representative PMs
TEACHER_MODES="${TEACHER_MODES:-client_all cluster_all client_conf cluster_conf}"

# Common model settings
FEDCD_FEXT_MODEL="${FEDCD_FEXT_MODEL:-SmallFExt}"
FEDCD_FEXT_DIM="${FEDCD_FEXT_DIM:-512}"
FEDCD_TEACHER_PROXY_DATASET="${FEDCD_TEACHER_PROXY_DATASET:-Cifar100}"
FEDCD_TEACHER_CORRECT_ONLY="${FEDCD_TEACHER_CORRECT_ONLY:-false}"
FEDCD_TEACHER_CONF_MIN="${FEDCD_TEACHER_CONF_MIN:-0.1}"
FEDCD_TEACHER_CONF_POWER="${FEDCD_TEACHER_CONF_POWER:-2.0}"

# PM-first inference fusion (PM 기본 + 필요할 때만 GM fallback)
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

# Confidence-filter settings for *_conf modes
CONF_CLIENT_TOPK="${CONF_CLIENT_TOPK:-5}"
CONF_CLUSTER_TOPK="${CONF_CLUSTER_TOPK:-3}"
CONF_ABSTAIN_THRESHOLD="${CONF_ABSTAIN_THRESHOLD:-0.3}"
CONF_TEACHER_ABSTAIN_THRESHOLD="${CONF_TEACHER_ABSTAIN_THRESHOLD:-0.5}"
CONF_MIN_ACTIVE="${CONF_MIN_ACTIVE:-2}"
CONF_CONSENSUS_MIN_RATIO="${CONF_CONSENSUS_MIN_RATIO:-0.6}"

# Combined score for "best PM-local + GM-global"
SCORE_PM_WEIGHT="${SCORE_PM_WEIGHT:-0.5}"
SCORE_GM_WEIGHT="${SCORE_GM_WEIGHT:-0.5}"

# Disable search early-stop by default to ensure full GLOBAL_ROUNDS runs
FEDCD_SEARCH_ENABLE="${FEDCD_SEARCH_ENABLE:-false}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
COMPARE_ROOT="${COMPARE_ROOT:-${ROOT_DIR}/compare}"
COMPARE_RUN_DIR="${COMPARE_ROOT}/${RUN_TS}"
LOG_DIR="${LOG_DIR:-${COMPARE_RUN_DIR}/console}"
EXP_DIR="${COMPARE_RUN_DIR}/experiments"
SUMMARY_CSV="${COMPARE_RUN_DIR}/summary.csv"
BEST_TXT="${COMPARE_RUN_DIR}/best.txt"
mkdir -p "${LOG_DIR}" "${EXP_DIR}"

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

collect_required_csv() {
  local search_root="$1"
  local stamp_file="$2"
  local dst_file="$3"

  local src
  src="$(latest_named_file_after "${search_root}" "${stamp_file}" "acc.csv")"
  if [ -z "${src}" ]; then
    src="$(latest_named_file "${search_root}" "acc.csv")"
  fi

  if [ -z "${src}" ]; then
    echo ""
    return 0
  fi
  cp "${src}" "${dst_file}"
  echo "${src}"
}

collect_optional_csv() {
  local search_root="$1"
  local stamp_file="$2"
  local file_name="$3"
  local dst_file="$4"

  local src
  src="$(latest_named_file_after "${search_root}" "${stamp_file}" "${file_name}")"
  if [ -z "${src}" ]; then
    src="$(latest_named_file "${search_root}" "${file_name}")"
  fi
  if [ -n "${src}" ]; then
    cp "${src}" "${dst_file}"
  fi
}

teacher_mode_to_cfg() {
  local mode="$1"
  local source=""
  local topk=""
  local abstain=""
  local teacher_abstain=""
  local min_active=""
  local consensus=""
  local desc=""
  case "${mode}" in
    client_all)
      source="client"
      topk="0"
      abstain="0.0"
      teacher_abstain="0.0"
      min_active="1"
      consensus="0.0"
      desc="all client PM teachers"
      ;;
    cluster_all)
      source="cluster"
      topk="0"
      abstain="0.0"
      teacher_abstain="0.0"
      min_active="1"
      consensus="0.0"
      desc="all cluster representative PM teachers"
      ;;
    client_conf)
      source="client"
      topk="${CONF_CLIENT_TOPK}"
      abstain="${CONF_ABSTAIN_THRESHOLD}"
      teacher_abstain="${CONF_TEACHER_ABSTAIN_THRESHOLD}"
      min_active="${CONF_MIN_ACTIVE}"
      consensus="${CONF_CONSENSUS_MIN_RATIO}"
      desc="confidence-filtered client PM teachers"
      ;;
    cluster_conf)
      source="cluster"
      topk="${CONF_CLUSTER_TOPK}"
      abstain="${CONF_ABSTAIN_THRESHOLD}"
      teacher_abstain="${CONF_TEACHER_ABSTAIN_THRESHOLD}"
      min_active="${CONF_MIN_ACTIVE}"
      consensus="${CONF_CONSENSUS_MIN_RATIO}"
      desc="confidence-filtered cluster PM teachers"
      ;;
    *)
      echo "[Error] Unknown teacher mode: ${mode}" >&2
      return 1
      ;;
  esac
  printf '%s|%s|%s|%s|%s|%s|%s\n' \
    "${source}" "${topk}" "${abstain}" "${teacher_abstain}" "${min_active}" "${consensus}" "${desc}"
}

calc_model_size_info() {
  local gm_hidden="$1"
  local pm_hidden="$2"
  python - "${ROOT_DIR}/FedCD-Core/system" "$gm_hidden" "$pm_hidden" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from flcore.trainmodel.models import VGG8, SmallFExt

gm_h = int(sys.argv[2])
pm_h = int(sys.argv[3])

def nparams(m):
    return sum(p.numel() for p in m.parameters())

fext = SmallFExt(out_dim=256)
gm = VGG8(num_classes=10, classifier_hidden=gm_h).classifier
pm = VGG8(num_classes=10, classifier_hidden=pm_h).classifier
base = VGG8(num_classes=10)

wrapper_params = nparams(fext) + nparams(gm) + nparams(pm)
base_params = nparams(base)
wrapper_mb = wrapper_params * 4 / (1024 ** 2)
base_mb = base_params * 4 / (1024 ** 2)
print(f"{wrapper_params},{base_params},{wrapper_mb:.6f},{base_mb:.6f}")
PY
}

extract_final_metrics() {
  local acc_csv="$1"
  python - "$acc_csv" <<'PY'
import csv
import sys

acc_csv = sys.argv[1]
rows = list(csv.DictReader(open(acc_csv)))
if not rows:
    print(",,,,,,,,")
    raise SystemExit(0)
r = rows[-1]
fields = [
    "round",
    "pm_local_test_acc",
    "gm_only_global_test_acc",
    "global_test_acc",
    "local_test_acc",
    "total_mb",
    "uplink_mb",
    "downlink_mb",
]
print(",".join(r.get(k, "") for k in fields))
PY
}

calc_score() {
  local pm_local="$1"
  local gm_global="$2"
  python - "$pm_local" "$gm_global" "$SCORE_PM_WEIGHT" "$SCORE_GM_WEIGHT" <<'PY'
import sys

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

pm = to_float(sys.argv[1], 0.0)
gm = to_float(sys.argv[2], 0.0)
w_pm = to_float(sys.argv[3], 0.5)
w_gm = to_float(sys.argv[4], 0.5)
print(f"{w_pm * pm + w_gm * gm:.6f}")
PY
}

cat > "${SUMMARY_CSV}" <<'CSV'
exp_id,teacher_mode,teacher_source,teacher_desc,gm_hidden,pm_hidden,sum_hidden,final_round,pm_local_final,gm_global_final,global_final,local_final,total_mb_last,uplink_mb_last,downlink_mb_last,wrapper_params,baseline_params,wrapper_model_mb,baseline_model_mb,combined_score,source_acc_path,console_log
CSV

exp_idx=0
for mode in ${TEACHER_MODES}; do
  cfg="$(teacher_mode_to_cfg "${mode}")"
  IFS='|' read -r teacher_source teacher_topk teacher_abstain teacher_teacher_abstain teacher_min_active teacher_consensus teacher_desc <<< "${cfg}"

  for split in ${MODEL_SPLITS}; do
    gm_hidden="${split%%:*}"
    pm_hidden="${split##*:}"
    if [ -z "${gm_hidden}" ] || [ -z "${pm_hidden}" ] || [ "${gm_hidden}" = "${split}" ]; then
      echo "[Warn] Skip malformed MODEL_SPLITS entry: ${split}"
      continue
    fi

    sum_hidden=$((gm_hidden + pm_hidden))
    exp_idx=$((exp_idx + 1))
    exp_id=$(printf "exp_%03d_%s_gm%s_pm%s" "${exp_idx}" "${mode}" "${gm_hidden}" "${pm_hidden}")
    exp_dir="${EXP_DIR}/${exp_id}"
    mkdir -p "${exp_dir}"

    echo ""
    echo "=================================================="
    echo "[Run ${exp_idx}] ${exp_id}"
    echo "teacher: ${teacher_desc}"
    echo "gm/pm hidden: ${gm_hidden}/${pm_hidden} (sum=${sum_hidden})"

    stamp_file="$(mktemp)"
    touch "${stamp_file}"
    console_log="${LOG_DIR}/${exp_id}.log"

    run_py_in_dir "${ROOT_DIR}/FedCD-Core" system/main.py \
      -go train \
      -algo FedCD \
      -dev cuda -did "${GPU_ID}" \
      -data "${DATASET}" \
      -ncl "${NUM_CLASSES}" \
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
      --fedcd_local_pm_only_objective true \
      --fedcd_gm_lr_scale 0.1 \
      --fedcd_gm_update_mode server_pm_teacher \
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
      --fedcd_pm_teacher_source "${teacher_source}" \
      --fedcd_pm_teacher_confidence_weight true \
      --fedcd_pm_teacher_confidence_min "${FEDCD_TEACHER_CONF_MIN}" \
      --fedcd_pm_teacher_confidence_power "${FEDCD_TEACHER_CONF_POWER}" \
      --fedcd_pm_teacher_ensemble_confidence true \
      --fedcd_pm_teacher_topk "${teacher_topk}" \
      --fedcd_pm_teacher_abstain_threshold "${teacher_abstain}" \
      --fedcd_pm_teacher_teacher_abstain_threshold "${teacher_teacher_abstain}" \
      --fedcd_pm_teacher_min_active_teachers "${teacher_min_active}" \
      --fedcd_pm_teacher_consensus_min_ratio "${teacher_consensus}" \
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
      --fedcd_search_enable "${FEDCD_SEARCH_ENABLE}" \
      --fedcd_search_min_rounds 8 \
      --fedcd_search_patience 6 \
      --fedcd_search_drop_patience 3 \
      --fedcd_search_drop_delta 0.003 \
      --fedcd_search_score_gm_weight 0.75 \
      --fedcd_search_score_pm_weight 0.25 \
      --fedcd_search_score_eps 0.0001 \
      --fedcd_search_min_pm_local_acc 0.55 \
      --fedcd_search_min_gm_global_acc 0.18 \
      --gm_model "VGG8W${gm_hidden}" \
      --pm_model "VGG8W${pm_hidden}" \
      --fext_model "${FEDCD_FEXT_MODEL}" \
      --fext_dim "${FEDCD_FEXT_DIM}" \
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
      -tth 10000 2>&1 | tee "${console_log}"

    acc_source="$(collect_required_csv "${ROOT_DIR}/FedCD-Core/logs/FedCD" "${stamp_file}" "${exp_dir}/acc.csv")"
    collect_optional_csv "${ROOT_DIR}/FedCD-Core/logs/FedCD" "${stamp_file}" "cluster_acc.csv" "${exp_dir}/cluster_acc.csv"
    collect_optional_csv "${ROOT_DIR}/FedCD-Core/logs/FedCD" "${stamp_file}" "usage.csv" "${exp_dir}/usage.csv"
    rm -f "${stamp_file}"

    if [ -z "${acc_source}" ]; then
      echo "[Warn] ${exp_id}: acc.csv not found. Skip summary row."
      continue
    fi
    printf '%s\n' "${acc_source}" > "${exp_dir}/source_acc_path.txt"

    metrics="$(extract_final_metrics "${exp_dir}/acc.csv")"
    IFS=',' read -r final_round pm_local_final gm_global_final global_final local_final total_mb_last uplink_mb_last downlink_mb_last <<< "${metrics}"

    size_info="$(calc_model_size_info "${gm_hidden}" "${pm_hidden}")"
    IFS=',' read -r wrapper_params baseline_params wrapper_model_mb baseline_model_mb <<< "${size_info}"

    combined_score="$(calc_score "${pm_local_final}" "${gm_global_final}")"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${exp_id}" \
      "${mode}" \
      "${teacher_source}" \
      "${teacher_desc}" \
      "${gm_hidden}" \
      "${pm_hidden}" \
      "${sum_hidden}" \
      "${final_round}" \
      "${pm_local_final}" \
      "${gm_global_final}" \
      "${global_final}" \
      "${local_final}" \
      "${total_mb_last}" \
      "${uplink_mb_last}" \
      "${downlink_mb_last}" \
      "${wrapper_params}" \
      "${baseline_params}" \
      "${wrapper_model_mb}" \
      "${baseline_model_mb}" \
      "${combined_score}" \
      "${acc_source}" \
      "${console_log}" >> "${SUMMARY_CSV}"
  done
done

python - "${SUMMARY_CSV}" "${BEST_TXT}" <<'PY'
import csv
import math
import sys

summary_csv = sys.argv[1]
best_txt = sys.argv[2]
rows = list(csv.DictReader(open(summary_csv)))

def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")

valid = [r for r in rows if not math.isnan(f(r, "pm_local_final")) and not math.isnan(f(r, "gm_global_final"))]
if not valid:
    msg = "No valid experiment rows in summary.csv"
    print(msg)
    with open(best_txt, "w") as w:
        w.write(msg + "\n")
    raise SystemExit(0)

best_pm = max(valid, key=lambda r: f(r, "pm_local_final"))
best_gm = max(valid, key=lambda r: f(r, "gm_global_final"))
best_score = max(valid, key=lambda r: f(r, "combined_score"))

lines = []
lines.append("Best by PM local:")
lines.append(
    f"  {best_pm['exp_id']} | pm_local={best_pm['pm_local_final']} | "
    f"gm_global={best_pm['gm_global_final']} | mode={best_pm['teacher_mode']} | "
    f"gm/pm={best_pm['gm_hidden']}/{best_pm['pm_hidden']}"
)
lines.append("Best by GM global:")
lines.append(
    f"  {best_gm['exp_id']} | pm_local={best_gm['pm_local_final']} | "
    f"gm_global={best_gm['gm_global_final']} | mode={best_gm['teacher_mode']} | "
    f"gm/pm={best_gm['gm_hidden']}/{best_gm['pm_hidden']}"
)
lines.append("Best combined (PM+GM):")
lines.append(
    f"  {best_score['exp_id']} | score={best_score['combined_score']} | "
    f"pm_local={best_score['pm_local_final']} | gm_global={best_score['gm_global_final']} | "
    f"mode={best_score['teacher_mode']} | gm/pm={best_score['gm_hidden']}/{best_score['pm_hidden']}"
)

text = "\n".join(lines)
print(text)
with open(best_txt, "w") as w:
    w.write(text + "\n")
PY

echo ""
echo "Done."
echo "Summary:"
echo "  ${SUMMARY_CSV}"
echo "Best-result note:"
echo "  ${BEST_TXT}"
