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
                     CUDA device(s) visible to all parallel jobs (default: 0).
  MAX_PARALLEL_JOBS  Maximum number of concurrent baseline jobs (default: 4).
  MODEL              Model architecture (default: VGG8).
  EVAL_SCENARIOS     Test views to evaluate (default: id,ood,mix).
  DATA_CHECK_ONLY    Validate data and stop before training (default: 0).
  AUTO_SUMMARIZE     Write the scenario summary after all jobs finish (default: true).
  SUMMARY_OUTPUT     Summary CSV path.

Each dataset-method pair trains once on the new class-pair training partition,
then evaluates the same checkpoint on the generated ID, OOD, and Mix views.
Scenario logs are written as eval_id/acc.csv, eval_ood/acc.csv, and
eval_mix/acc.csv below each experiment directory.
EOF
}

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$REPO_DIR/FedCD-Baseline}"
RUNNER="$BASELINE_DIR/run_fedprism_idoodmix_vgg8.sh"
WORKSPACE_DIR="$(cd "$REPO_DIR/.." && pwd)"

FEDCD_PYTHON_BIN="${FEDCD_PYTHON:-$(command -v python 2>/dev/null || true)}"
FL_DATA_ROOT="${FL_DATA_ROOT:-$WORKSPACE_DIR/fl_data}"
RUN_SEED="${1:-${SEED:-1}}"
METHODS_ARG="${2:-${METHODS_CSV:-}}"
DEFAULT_METHODS='FedAvg,FedProx,FedCross,FedBN,FedALA,FedAS,pFedMe,cwFedAvg,FedDST,PMOE_FedPer,FedCP,DualFed'
SELECTED_METHODS="${METHODS_ARG:-$DEFAULT_METHODS}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-id,ood,mix}"

if [[ -z "$FEDCD_PYTHON_BIN" || ! -x "$FEDCD_PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter not found: ${FEDCD_PYTHON_BIN:-<empty>}" >&2
  echo '        Activate the pfllib environment or set FEDCD_PYTHON.' >&2
  exit 2
fi
if [[ ! "$RUN_SEED" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SEED must be a non-negative integer: $RUN_SEED" >&2
  exit 2
fi
for required_path in "$BASELINE_DIR" "$RUNNER" "$FL_DATA_ROOT"; do
  if [[ ! -e "$required_path" ]]; then
    echo "[ERROR] Required path not found: $required_path" >&2
    exit 2
  fi
done

validate_dataset() {
  local dataset_name="$1"
  local source_name="$2"
  "$FEDCD_PYTHON_BIN" - "$FL_DATA_ROOT/$dataset_name" "$source_name" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
source_name = sys.argv[2]
config_path = root / 'config.json'
if not config_path.is_file():
    raise SystemExit(f'missing config.json: {config_path}')
with config_path.open(encoding='utf-8') as handle:
    config = json.load(handle)
expected = {
    'schema': 'fedprism_id_ood_mix_v1',
    'dataset': source_name,
    'num_clients': 50,
    'num_classes': 10,
}
errors = [
    f'{key}={config.get(key)!r} (expected {value!r})'
    for key, value in expected.items()
    if config.get(key) != value
]
for split in ('train', 'test/id', 'test/ood', 'test/mix'):
    split_dir = root / split
    actual_ids = sorted(
        int(path.stem) for path in split_dir.glob('*.npz') if path.stem.isdigit()
    ) if split_dir.is_dir() else []
    if actual_ids != list(range(50)):
        errors.append(f'{split} client files are incomplete')
if not (root / 'test' / 'pool.npz').is_file():
    errors.append('test/pool.npz is missing')
if errors:
    raise SystemExit('; '.join(errors))
PY
}

echo "[DATA] Checking FedPRISM ID/OOD/Mix datasets under: $FL_DATA_ROOT"
validate_dataset 'Cifar10_fedprism_idoodmix_nc50' 'Cifar10'
validate_dataset 'FashionMNIST_fedprism_idoodmix_nc50' 'FashionMNIST'
echo '[DATA] Both generated datasets passed the baseline input checks.'

if [[ "${DATA_CHECK_ONLY:-0}" =~ ^(1|true|yes|y|on)$ ]]; then
  echo '[DATA] Validation completed; DATA_CHECK_ONLY is enabled.'
  exit 0
fi

export FEDCD_PYTHON="$FEDCD_PYTHON_BIN"
export FL_DATA_ROOT
export DATASETS='Cifar10,FashionMNIST'
export ALGORITHMS="$SELECTED_METHODS"
export EVAL_SCENARIOS
export MODEL="${MODEL:-VGG8}"
export GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
export LR="${LR:-0.005}"
export LBS="${LBS:-128}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
export JOIN_RATIO="${JOIN_RATIO:-1.0}"
export TIMES='1'
export SEED="$RUN_SEED"
export NUM_CLIENTS='50'
export NUM_CLASSES='10'
export EVAL_GAP="${EVAL_GAP:-1}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export AUTO_SUMMARIZE="${AUTO_SUMMARIZE:-true}"
export SUMMARY_TARGET_RUNS="${SUMMARY_TARGET_RUNS:-1}"
export SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$BASELINE_DIR/fedprism_idoodmix_result.csv}"

echo '=========================================================='
echo '[RQ1] Train-once ID/OOD/Mix baseline evaluation'
echo "[RQ1] methods=$ALGORITHMS"
echo "[RQ1] datasets=$DATASETS"
echo "[RQ1] scenarios=$EVAL_SCENARIOS"
echo "[RQ1] seed=$SEED, times=$TIMES"
echo "[RQ1] max_parallel_jobs=$MAX_PARALLEL_JOBS"
echo "[RQ1] summary=$SUMMARY_OUTPUT"
echo '=========================================================='

exec bash "$RUNNER"
