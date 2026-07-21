#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash shell/rq1_baselines/run_rq1_baselines_parallel4.sh [SEED] [METHODS]

Arguments:
  SEED     Single experiment seed (default: 1).
  METHODS  Optional comma-separated methods. By default, run all RQ1 baselines.

Environment variables:
  FEDCD_PYTHON       Python interpreter (default: command -v python).
  FL_DATA_ROOT       Shared FL dataset root (default: sibling fl_data directory).
  CUDA_VISIBLE_DEVICES
                     CUDA device(s) visible to all four parallel jobs (default: 0).
  MAX_PARALLEL_JOBS  Maximum number of concurrent baseline jobs (default: 4).
  DATA_SEED          SplitGP dataset generation seed (default: 1).
  MODEL              Model architecture (default: VGG8).
  DATA_CHECK_ONLY    Validate/generate data and stop before training (default: 0).

Each dataset-method pair trains once on rho=0.0 and evaluates
rho={0.0,0.2,0.4,0.6,0.8} from the same trained checkpoint. Each rho-specific
acc.csv contains mixed, ID, and client-level OOD accuracy columns.
The data partition uses the fixed class pairs {0,1}, {2,3}, ..., {8,9}.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$REPO_DIR/FedCD-Baseline}"
RUNNER="$BASELINE_DIR/run_splitgp_multirho_resnet10.sh"
GENERATOR="$REPO_DIR/tools/generate_splitgp_rho_data.py"
WORKSPACE_DIR="$(cd "$REPO_DIR/.." && pwd)"

FEDCD_PYTHON_BIN="${FEDCD_PYTHON:-$(command -v python 2>/dev/null || true)}"
FL_DATA_ROOT="${FL_DATA_ROOT:-$WORKSPACE_DIR/fl_data}"
RUN_SEED="${1:-${SEED:-1}}"
METHODS_ARG="${2:-${METHODS_CSV:-}}"
DATA_SEED="${DATA_SEED:-1}"

DEFAULT_METHODS="FedAvg,FedProx,FedCross,FedBN,FedALA,FedAS,pFedMe,cwFedAvg,FedDST,PMOE_FedPer,FedCP,DualFed"
SELECTED_METHODS="${METHODS_ARG:-$DEFAULT_METHODS}"
RQ1_RHOS="0.0,0.2,0.4,0.6,0.8"

if [[ -z "$FEDCD_PYTHON_BIN" || ! -x "$FEDCD_PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: ${FEDCD_PYTHON_BIN:-<empty>}" >&2
  echo "        Activate the pfllib environment or set FEDCD_PYTHON." >&2
  exit 2
fi
if [[ ! "$RUN_SEED" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SEED must be a non-negative integer: $RUN_SEED" >&2
  exit 2
fi
if [[ ! "$DATA_SEED" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] DATA_SEED must be a non-negative integer: $DATA_SEED" >&2
  exit 2
fi
for required_path in "$BASELINE_DIR" "$RUNNER" "$GENERATOR"; do
  if [[ ! -e "$required_path" ]]; then
    echo "[ERROR] Required path not found: $required_path" >&2
    exit 2
  fi
done

mkdir -p "$FL_DATA_ROOT"
export FL_DATA_ROOT

validate_existing_dataset() {
  local dataset_dir="$1"
  local display_name="$2"
  local rho="$3"

  "$FEDCD_PYTHON_BIN" - "$dataset_dir" "$display_name" "$rho" "$DATA_SEED" <<'PY'
import json
import sys
from pathlib import Path

dataset_dir = Path(sys.argv[1])
display_name = sys.argv[2]
rho = float(sys.argv[3])
seed = int(sys.argv[4])
config_path = dataset_dir / 'config.json'

if not config_path.is_file():
    raise SystemExit('missing config.json')

with config_path.open('r', encoding='utf-8') as file_obj:
    config = json.load(file_obj)

expected = {
    'dataset_source': display_name,
    'num_clients': 50,
    'splitgp_rho': rho,
    'splitgp_partition_mode': 'class_pair',
    'splitgp_num_shards': 100,
    'splitgp_shards_per_client': 2,
    'splitgp_test_samples_per_client': 1000,
    'seed': seed,
}

errors = []
for key, expected_value in expected.items():
    actual = config.get(key)
    if isinstance(expected_value, float):
        valid = actual is not None and abs(float(actual) - expected_value) < 1e-12
    else:
        valid = actual == expected_value
    if not valid:
        errors.append(f'{key}={actual!r} (expected {expected_value!r})')

for split in ('train', 'test'):
    split_dir = dataset_dir / split
    actual_ids = sorted(
        int(path.stem) for path in split_dir.glob('*.npz') if path.stem.isdigit()
    ) if split_dir.is_dir() else []
    if actual_ids != list(range(50)):
        errors.append(f'{split} client files are incomplete')

if errors:
    print('; '.join(errors), file=sys.stderr)
    raise SystemExit(1)
PY
}

dataset_keys=(cifar10 fashionmnist)
dataset_names=(Cifar10 FashionMNIST)
rhos=(0.0 0.2 0.4 0.6 0.8)
missing_datasets=()
invalid_datasets=()

echo "[DATA] Checking SplitGP RQ1 datasets under: $FL_DATA_ROOT"
for dataset_idx in "${!dataset_keys[@]}"; do
  display_name="${dataset_names[$dataset_idx]}"
  for rho in "${rhos[@]}"; do
    dataset_name="${display_name}_splitgp_pat_rho${rho}_nc50"
    dataset_dir="$FL_DATA_ROOT/$dataset_name"
    if [[ ! -d "$dataset_dir" ]]; then
      missing_datasets+=("$dataset_name")
      continue
    fi
    if ! validation_error="$(validate_existing_dataset "$dataset_dir" "$display_name" "$rho" 2>&1)"; then
      invalid_datasets+=("$dataset_name: $validation_error")
    fi
  done
done

if [[ "${#invalid_datasets[@]}" -gt 0 ]]; then
  echo "[ERROR] Existing datasets do not match the RQ1 configuration:" >&2
  printf '  - %s\n' "${invalid_datasets[@]}" >&2
  echo "        Existing data was not overwritten." >&2
  exit 2
fi

if [[ "${#missing_datasets[@]}" -gt 0 ]]; then
  echo "[DATA] Missing ${#missing_datasets[@]} dataset(s); generating the complete RQ1 set."
  printf '  - %s\n' "${missing_datasets[@]}"
  "$FEDCD_PYTHON_BIN" "$GENERATOR" \
    --datasets cifar10 fashionmnist \
    --rhos 0.0 0.2 0.4 0.6 0.8 \
    --num-clients 50 \
    --num-shards 100 \
    --shards-per-client 2 \
    --test-samples-per-client 1000 \
    --partition-mode class_pair \
    --seed "$DATA_SEED" \
    --output-root "$FL_DATA_ROOT"
else
  echo "[DATA] All required datasets exist and match the RQ1 configuration."
fi

# Verify again after generation so training never starts on a partial dataset.
for dataset_idx in "${!dataset_keys[@]}"; do
  display_name="${dataset_names[$dataset_idx]}"
  for rho in "${rhos[@]}"; do
    dataset_name="${display_name}_splitgp_pat_rho${rho}_nc50"
    validate_existing_dataset "$FL_DATA_ROOT/$dataset_name" "$display_name" "$rho"
  done
done

if [[ "${DATA_CHECK_ONLY:-0}" =~ ^(1|true|yes|y|on)$ ]]; then
  echo "[DATA] Validation completed; DATA_CHECK_ONLY is enabled."
  exit 0
fi

export FEDCD_PYTHON="$FEDCD_PYTHON_BIN"
export DATASETS="Cifar10,FashionMNIST"
export ALGORITHMS="$SELECTED_METHODS"
export TRAIN_RHO="0.0"
export EVAL_RHOS="$RQ1_RHOS"
export ENABLE_MULTI_RHO_EVAL="true"
export MODEL="${MODEL:-VGG8}"
export GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
export LR="${LR:-0.005}"
export LBS="${LBS:-128}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
export JOIN_RATIO="${JOIN_RATIO:-1.0}"
export TIMES="1"
export SEED="$RUN_SEED"
export NUM_CLIENTS="50"
export NUM_CLASSES="10"
export EVAL_GAP="${EVAL_GAP:-1}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=========================================================="
echo "[RQ1] One-run baseline evaluation"
echo "[RQ1] methods=$ALGORITHMS"
echo "[RQ1] datasets=$DATASETS"
echo "[RQ1] train_rho=$TRAIN_RHO"
echo "[RQ1] eval_rhos=$EVAL_RHOS"
echo "[RQ1] seed=$SEED, times=$TIMES"
echo "[RQ1] max_parallel_jobs=$MAX_PARALLEL_JOBS"
echo "[RQ1] metrics=mixed_acc,id_acc,ood_acc"
echo "=========================================================="

exec bash "$RUNNER"
