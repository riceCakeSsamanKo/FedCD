#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_PYTHON='/home1/irteam/.conda/envs/pfllib/bin/python'
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
export TIMES='1'
export NUM_CLIENTS="${NUM_CLIENTS:-50}"
export NUM_CLASSES="${NUM_CLASSES:-10}"
export DATASETS="${DATASETS:-Cifar10,FashionMNIST}"
export EVAL_SCENARIOS="${EVAL_SCENARIOS:-id,ood,mix}"
export ALGORITHMS="${ALGORITHMS:-FedAvg,FedProx,FedCross,FedBN,FedALA,FedAS,pFedMe,cwFedAvg,FedDST,PMOE_FedPer,FedCP,DualFed}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${PARALLEL_PROCESSES:-5}}"

SEEDS_CSV="${SEEDS:-1,2,3}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$SCRIPT_DIR/fedprism_idoodmix_result.csv}"
IFS=',' read -r -a seeds <<< "$SEEDS_CSV"

echo '=========================================================='
echo '[ALL] FedPRISM ID/OOD/Mix baseline experiments'
echo "[ALL] Python: $FEDCD_PYTHON"
echo "[ALL] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[ALL] datasets=$DATASETS"
echo "[ALL] scenarios=$EVAL_SCENARIOS"
echo "[ALL] algorithms=$ALGORITHMS"
echo "[ALL] seeds=$SEEDS_CSV"
echo "[ALL] parallel jobs=$MAX_PARALLEL_JOBS"
echo '=========================================================='

for seed_raw in "${seeds[@]}"; do
  seed="$(echo "$seed_raw" | xargs)"
  [[ -n "$seed" ]] || continue
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid seed: $seed" >&2
    exit 2
  fi
  echo "[ALL] Starting seed=$seed"
  SEED="$seed" AUTO_SUMMARIZE=false \
    bash "$SCRIPT_DIR/run_fedprism_idoodmix_vgg8.sh"
done

"$FEDCD_PYTHON" "$SCRIPT_DIR/tools/summarize_fedprism_scenarios.py" \
  --logs-root "$SCRIPT_DIR/logs" \
  --datasets "$DATASETS" \
  --methods "$ALGORITHMS" \
  --model "$MODEL" \
  --clients "$NUM_CLIENTS" \
  --scenarios "$EVAL_SCENARIOS" \
  --required-round "$((GLOBAL_ROUNDS + 1))" \
  --target-runs "${SUMMARY_TARGET_RUNS:-${#seeds[@]}}" \
  --scale percent \
  --decimals 2 \
  --output-csv "$SUMMARY_OUTPUT"

echo "[ALL] Done. Summary: $SUMMARY_OUTPUT"
