#!/usr/bin/env bash
set -euo pipefail

cd /home1/irteam/workspace/FedCD/FedCD-Baseline

export FEDCD_PYTHON=/home1/irteam/.conda/envs/pfllib/bin/python
export FL_DATA_ROOT=/home1/irteam/workspace/fl_data
export LD_LIBRARY_PATH=/home1/irteam/.conda/envs/pfllib/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DEVICE_ID=${DEVICE_ID:-0}

export TARGET_RUNS=${TARGET_RUNS:-3}
export MAX_PARALLEL=${MAX_PARALLEL:-2}

bash run_splitgp_rho_baselines_vgg8_3runs_parallel2.sh
