#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SYSTEM_DIR="$REPO_ROOT/FedCD-Baseline/system"
PYTHON_BIN="${FEDCD_PYTHON:-/data/miniconda3/envs/pfllib/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="FashionMNIST_dir0.5_nc20"
SCENARIO="dir0.5_nc20"
MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="${DEVICE_ID:-0}"
GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-100}"
LR="${LR:-0.005}"
LBS="${LBS:-128}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
JOIN_RATIO="${JOIN_RATIO:-1.0}"
NUM_CLASSES="10"
NUM_CLIENTS="20"
TIMES="1"
ETA="${ETA:-1.0}"
RAND_PERCENT="${RAND_PERCENT:-80}"
LAYER_IDX="${LAYER_IDX:-2}"
SEEDS=(${SEEDS:-0 1 2 3 4})

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python interpreter not found or not executable: $PYTHON_BIN"
    echo "[HINT] Set FEDCD_PYTHON=/path/to/python if needed."
    exit 1
fi

if [[ ! -d "$SYSTEM_DIR" ]]; then
    echo "[ERROR] System directory not found: $SYSTEM_DIR"
    exit 1
fi

choose_fl_data_root() {
    if [[ -n "${FL_DATA_ROOT:-}" ]]; then
        echo "$FL_DATA_ROOT"
        return 0
    fi

    local sibling_root
    sibling_root="$(cd "$REPO_ROOT/.." && pwd)/fl_data"
    local repo_root_data="$REPO_ROOT/fl_data"

    if [[ -d "$sibling_root/$DATASET" ]]; then
        echo "$sibling_root"
        return 0
    fi
    if [[ -d "$repo_root_data/$DATASET" ]]; then
        echo "$repo_root_data"
        return 0
    fi

    echo "$sibling_root"
}

export FL_DATA_ROOT="$(choose_fl_data_root)"

if [[ ! -d "$FL_DATA_ROOT/$DATASET/train" || ! -d "$FL_DATA_ROOT/$DATASET/test" ]]; then
    echo "[ERROR] Required dataset not found: $FL_DATA_ROOT/$DATASET"
    echo "[HINT] Generate the shared FedCCM/FedCD data first, or set FL_DATA_ROOT explicitly."
    exit 1
fi

DATE_STR="$(date -u +%Y%m%d)"
TIME_STR="$(date -u +%H%M%S)"
QUEUE_ROOT="$SCRIPT_DIR/batch_runs/fedala_vgg8_fashionmnist_dir05_nc20_5seeds/date_${DATE_STR}/time_${TIME_STR}"
mkdir -p "$QUEUE_ROOT"

STATUS_CSV="$QUEUE_ROOT/status.csv"
echo "algorithm,scenario,dataset,seed,status,exit_code,start_utc,end_utc,fl_data_root,goal" > "$STATUS_CSV"

echo "[INFO] Queue root: $QUEUE_ROOT"
echo "[INFO] Using python: $PYTHON_BIN"
echo "[INFO] FL_DATA_ROOT: $FL_DATA_ROOT"
echo "[INFO] Dataset: $DATASET"
echo "[INFO] Seeds: ${SEEDS[*]}"
echo

for seed in "${SEEDS[@]}"; do
    goal="FedALA_${SCENARIO}_seed${seed}_${DATE_STR}_${TIME_STR}"
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit_code=0
    status="ok"

    echo "=========================================================="
    echo "[START] algo=FedALA seed=$seed dataset=$DATASET"
    echo "[CONFIG] model=$MODEL rounds=$GLOBAL_ROUNDS lr=$LR lbs=$LBS ls=$LOCAL_EPOCHS jr=$JOIN_RATIO"
    echo "[EXTRA] eta=$ETA rand_percent=$RAND_PERCENT layer_idx=$LAYER_IDX"
    echo "=========================================================="

    "$PYTHON_BIN" -u "$SYSTEM_DIR/main.py" \
        -data "$DATASET" \
        -ncl "$NUM_CLASSES" \
        -m "$MODEL" \
        -algo FedALA \
        -gr "$GLOBAL_ROUNDS" \
        -lr "$LR" \
        -lbs "$LBS" \
        -ls "$LOCAL_EPOCHS" \
        -nc "$NUM_CLIENTS" \
        -jr "$JOIN_RATIO" \
        -t "$TIMES" \
        -go "$goal" \
        -dev "$DEVICE" \
        -did "$DEVICE_ID" \
        --seed "$seed" \
        -et "$ETA" \
        -s "$RAND_PERCENT" \
        -p "$LAYER_IDX"
    exit_code=$?

    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ $exit_code -ne 0 ]]; then
        status="failed"
        echo "[FAIL] algo=FedALA seed=$seed exit_code=$exit_code"
    else
        echo "[DONE] algo=FedALA seed=$seed"
    fi

    echo "FedALA,${SCENARIO},${DATASET},${seed},${status},${exit_code},${start_utc},${end_utc},${FL_DATA_ROOT},${goal}" >> "$STATUS_CSV"
    echo

    if [[ $exit_code -ne 0 ]]; then
        echo "[INFO] Stopping after failed seed. Status CSV: $STATUS_CSV"
        exit "$exit_code"
    fi
done

echo "[INFO] FedALA FashionMNIST dir0.5 NC20 5-seed queue finished."
echo "[INFO] Status CSV: $STATUS_CSV"
