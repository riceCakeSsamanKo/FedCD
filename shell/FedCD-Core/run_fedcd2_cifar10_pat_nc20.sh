#!/usr/bin/env bash
set -euo pipefail

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

GM_MODEL="${GM_MODEL:-VGG8}"
PM_MODEL="${PM_MODEL:-VGG8W224}"
NUM_CLUSTERS="${NUM_CLUSTERS:-5}"
CLUSTER_PERIOD="${CLUSTER_PERIOD:-5}"
GM_PERIOD="${GM_PERIOD:-1}"
PLOCAL_EPOCHS="${PLOCAL_EPOCHS:-1}"
FEDCD2_CLUSTER_WARMUP_ROUNDS="${FEDCD2_CLUSTER_WARMUP_ROUNDS:-5}"
FEDCD2_BETA="${FEDCD2_BETA:-0.5}"
FEDCD2_FNC_WEIGHT="${FEDCD2_FNC_WEIGHT:-0.05}"
FEDCD2_LOCAL_PM_WEIGHT="${FEDCD2_LOCAL_PM_WEIGHT:-0.5}"
FEDCD2_CLUSTER_PM_WEIGHT="${FEDCD2_CLUSTER_PM_WEIGHT:-0.5}"
FEDCD2_PM_FEATURE_DIM="${FEDCD2_PM_FEATURE_DIM:-128}"
FEDCD2_FNC_DIM="${FEDCD2_FNC_DIM:-128}"
FEDCD2_PM_VGG_WIDTH_RATIO="${FEDCD2_PM_VGG_WIDTH_RATIO:-0.25}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "FedCD2 run"
echo "dataset=${DATASET} nc=${NUM_CLIENTS} gr=${GLOBAL_ROUNDS} ls=${LOCAL_EPOCHS} lr=${LR} lbs=${BATCH_SIZE}"
echo "model: GM=${GM_MODEL}, PM=${PM_MODEL}"
echo "cluster: k=${NUM_CLUSTERS} period=${CLUSTER_PERIOD} warmup=${FEDCD2_CLUSTER_WARMUP_ROUNDS}"
echo "pm-sync: local=${FEDCD2_LOCAL_PM_WEIGHT} cluster=${FEDCD2_CLUSTER_PM_WEIGHT}"
echo "=========================================================="

cd "${ROOT_DIR}"
CONDA_NO_PLUGINS=true conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u system/main.py \
    -go "${GOAL}" \
    -algo FedCD2 \
    -dev cuda -did "${GPU_ID}" \
    -data "${DATASET}" \
    -ncl "${NUM_CLASSES}" \
    -m "${GM_MODEL}" \
    --gm_model "${GM_MODEL}" \
    --pm_model "${PM_MODEL}" \
    -lbs "${BATCH_SIZE}" \
    -lr "${LR}" \
    -ld false \
    -ldg 0.99 \
    -gr "${GLOBAL_ROUNDS}" \
    -ls "${LOCAL_EPOCHS}" \
    -jr "${JOIN_RATIO}" \
    -nc "${NUM_CLIENTS}" \
    -t 1 \
    --num_clusters "${NUM_CLUSTERS}" \
    --cluster_period "${CLUSTER_PERIOD}" \
    --global_period "${GM_PERIOD}" \
    --eval_gap 1 \
    --plocal_epochs "${PLOCAL_EPOCHS}" \
    --fedcd2_cluster_warmup_rounds "${FEDCD2_CLUSTER_WARMUP_ROUNDS}" \
    --fedcd2_beta "${FEDCD2_BETA}" \
    --fedcd2_fnc_weight "${FEDCD2_FNC_WEIGHT}" \
    --fedcd2_local_pm_weight "${FEDCD2_LOCAL_PM_WEIGHT}" \
    --fedcd2_cluster_pm_weight "${FEDCD2_CLUSTER_PM_WEIGHT}" \
    --fedcd2_pm_feature_dim "${FEDCD2_PM_FEATURE_DIM}" \
    --fedcd2_fnc_dim "${FEDCD2_FNC_DIM}" \
    --fedcd2_pm_vgg_width_ratio "${FEDCD2_PM_VGG_WIDTH_RATIO}"
