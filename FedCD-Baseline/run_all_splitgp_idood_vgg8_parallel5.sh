#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_PYTHON="/home1/irteam/.conda/envs/pfllib/bin/python"
if [[ -n "${FEDCD_PYTHON:-}" ]]; then
  export FEDCD_PYTHON
elif [[ -x "$DEFAULT_PYTHON" ]]; then
  export FEDCD_PYTHON="$DEFAULT_PYTHON"
else
  export FEDCD_PYTHON="$(command -v python)"
fi

PYTHON_PREFIX="$(cd "$(dirname "$FEDCD_PYTHON")/.." && pwd)"
if [[ -d "$PYTHON_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

export FL_DATA_ROOT="${FL_DATA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)/fl_data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE_ID="${DEVICE_ID:-0}"
export DEVICE="${DEVICE:-cuda}"

export MODEL="${MODEL:-VGG8}"
export GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
export LR="${LR:-0.005}"
export LBS="${LBS:-128}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
export JOIN_RATIO="${JOIN_RATIO:-1.0}"
export TIMES="${TIMES:-1}"
export NUM_CLIENTS="${NUM_CLIENTS:-50}"

export DATASETS="${DATASETS:-Cifar10,FashionMNIST}"
export RHOS="${RHOS:-0.0,0.2,0.4,0.6,0.8}"
export EVAL_RHOS="${EVAL_RHOS:-0.0,0.2,0.4,0.6,0.8}"
export REQUIRE_ID_OOD=1

export TARGET_RUNS="${TARGET_RUNS:-3}"
PARALLEL_PROCESSES="${PARALLEL_PROCESSES:-5}"
export MAX_PARALLEL="$PARALLEL_PROCESSES"
export MAX_PARALLEL_JOBS="$PARALLEL_PROCESSES"
export MAX_PARALLEL_RHOS="$PARALLEL_PROCESSES"

COMMON_METHODS_CSV="${COMMON_METHODS:-FedAvg,FedAS,FedProx,FedBN,FedALA,FedCross,pFedMe,cwFedAvg}"
SPECIAL_METHODS_CSV="${SPECIAL_METHODS:-FedDST,FedMoE,PMOE_FedPer,FedCP,DualFed}"
SEEDS_CSV="${SEEDS:-1,2,3}"

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  local old_ifs="$IFS"
  IFS=','
  read -r -a out_ref <<< "$csv"
  IFS="$old_ifs"
}

split_csv "$COMMON_METHODS_CSV" common_methods
split_csv "$SPECIAL_METHODS_CSV" special_methods
split_csv "$SEEDS_CSV" seeds

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
}

require_path "$FEDCD_PYTHON"
require_path "$FL_DATA_ROOT"
require_path "$SCRIPT_DIR/run_splitgp_rho_baselines_vgg8_3runs_parallel2.sh"

echo "=========================================================="
echo "[ALL] SplitGP ID/OOD baseline experiments"
echo "[ALL] Python: $FEDCD_PYTHON"
echo "[ALL] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[ALL] GPU device_id: $DEVICE_ID / CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[ALL] datasets=$DATASETS"
echo "[ALL] train rhos=$RHOS"
echo "[ALL] eval rhos=$EVAL_RHOS"
echo "[ALL] target_runs=$TARGET_RUNS seeds=$SEEDS_CSV"
echo "[ALL] max parallel processes per method=$PARALLEL_PROCESSES"
echo "[ALL] ID/OOD logging is required: REQUIRE_ID_OOD=$REQUIRE_ID_OOD"
echo "=========================================================="

run_common_method() {
  local method="$1"
  echo
  echo "=========================================================="
  echo "[METHOD] $method via 3-run scheduler"
  echo "=========================================================="
  ALGORITHMS="$method" \
    bash "$SCRIPT_DIR/run_splitgp_rho_baselines_vgg8_3runs_parallel2.sh"
}

run_seeded_script() {
  local method="$1"
  local script="$2"
  require_path "$SCRIPT_DIR/$script"
  for seed in "${seeds[@]}"; do
    seed="$(echo "$seed" | xargs)"
    [[ -n "$seed" ]] || continue
    echo
    echo "=========================================================="
    echo "[METHOD] $method seed=$seed"
    echo "=========================================================="
    SEED="$seed" bash "$SCRIPT_DIR/$script"
  done
}

for method in "${common_methods[@]}"; do
  method="$(echo "$method" | xargs)"
  [[ -n "$method" ]] || continue
  run_common_method "$method"
done

for method in "${special_methods[@]}"; do
  method="$(echo "$method" | xargs)"
  [[ -n "$method" ]] || continue
  case "$method" in
    FedDST)
      run_seeded_script "$method" "run_feddst_splitgp_rho_vgg8.sh"
      ;;
    FedMoE)
      run_seeded_script "$method" "run_fedmoe_splitgp_rho_vgg8.sh"
      ;;
    PMOE_FedPer)
      run_seeded_script "$method" "run_pmoe_fedper_splitgp_rho_vgg8.sh"
      ;;
    FedCP)
      run_seeded_script "$method" "run_fedcp_splitgp_rho_vgg8.sh"
      ;;
    DualFed)
      run_seeded_script "$method" "run_dualfed_splitgp_rho_vgg8.sh"
      ;;
    *)
      echo "[ERROR] Unknown special method: $method" >&2
      exit 1
      ;;
  esac
done

echo
echo "=========================================================="
echo "[ALL] Done. Logs are under: $SCRIPT_DIR/logs"
echo "[ALL] Every new acc.csv should include id_test_acc and ood_test_acc columns."
echo "=========================================================="
