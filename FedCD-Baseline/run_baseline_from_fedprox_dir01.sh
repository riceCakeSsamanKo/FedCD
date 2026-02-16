#!/bin/bash

# Resume baseline experiments from a specific checkpoint:
# Start from (FedProx, dir0.1_nc20), then run all remaining combinations
# in the original loop order.

set +e

# Experiment settings
MODEL="VGG16"
GR=100
LR=0.01
LBS=10
LS=1
DEVICE="cuda"
DEVICE_ID="0"

# Algorithms to test (original order)
algorithms=("FedAvg" "FedProx" "FedAS" "Local" "FedKD")

# Scenarios available in fl_data (original order)
scenarios=(
    "pat_nc20" "dir0.1_nc20" "dir0.5_nc20" "dir1.0_nc20"
    "pat_nc50" "dir0.1_nc50" "dir0.5_nc50" "dir1.0_nc50"
)

# Resume checkpoint
START_ALGO="FedProx"
START_SCENARIO="dir0.1_nc20"

found_start=false

# Go to system directory
cd system || exit 1

for algo in "${algorithms[@]}"; do
    for scenario in "${scenarios[@]}"; do
        if [ "$found_start" = false ]; then
            if [[ "$algo" == "$START_ALGO" && "$scenario" == "$START_SCENARIO" ]]; then
                found_start=true
            else
                continue
            fi
        fi

        DATASET="Cifar10_$scenario"

        if [[ $scenario =~ nc([0-9]+) ]]; then
            nc=${BASH_REMATCH[1]}
        else
            nc=20
        fi

        GOAL="${algo}_${scenario}"
        echo "=========================================================="
        echo "Running $algo for Scenario: $scenario (Clients: $nc)"
        echo "=========================================================="

        python -u main.py \
            -data "$DATASET" \
            -m "$MODEL" \
            -algo "$algo" \
            -gr "$GR" \
            -lr "$LR" \
            -lbs "$LBS" \
            -ls "$LS" \
            -nc "$nc" \
            -go "$GOAL" \
            -dev "$DEVICE" \
            -did "$DEVICE_ID"
    done
done

if [ "$found_start" = false ]; then
    echo "Start checkpoint not found: algo=$START_ALGO, scenario=$START_SCENARIO"
    exit 1
fi

echo "Baseline experiments completed from ${START_ALGO}/${START_SCENARIO} onward."
