#!/bin/bash

set -euo pipefail

cd /home/mulsoap0504/FedCD/FedCD-Baseline/system
export MPLCONFIGDIR=/tmp/mpl
PY=/data/miniconda3/envs/pfllib/bin/python

for scenario in pat_nc20 pat_nc50; do
    if [[ "$scenario" == *_nc50 ]]; then
        nc=50
    else
        nc=20
    fi

    dataset="Cifar10_${scenario}"
    goal="LG-FedAvg_${scenario}_probe_$(date -u +%Y%m%d_%H%M%S)"
    echo "=========================================================="
    echo "[START] scenario=$scenario dataset=$dataset nc=$nc"
    echo "=========================================================="
    "$PY" -u main.py \
        -data "$dataset" \
        -ncl 10 \
        -m VGG8 \
        -algo LG-FedAvg \
        -gr 5 \
        -lr 0.005 \
        -lbs 128 \
        -ls 2 \
        -nc "$nc" \
        -jr 1.0 \
        -t 1 \
        -go "$goal" \
        -dev cuda \
        -did 0
done

echo "[DONE] LG-FedAvg probe finished"
