#!/bin/bash

# Common runtime settings
MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="0"
ALGORITHM="Local"

# Scenarios available in fl_data
scenarios=(
    "pat_nc20" "dir0.1_nc20" "dir0.5_nc20" "dir1.0_nc20"
    "pat_nc50" "dir0.1_nc50" "dir0.5_nc50" "dir1.0_nc50"
)

# Go to system directory
cd system || exit 1

for scenario in "${scenarios[@]}"; do
    DATASET="Cifar10_$scenario"

    if [[ $scenario =~ nc([0-9]+) ]]; then
        nc=${BASH_REMATCH[1]}
    else
        nc=20
    fi

    GOAL="${ALGORITHM}_${scenario}"

    # Common hyperparameters aligned with FedCD defaults.
    GR=100
    LR=0.005
    LBS=128
    LS=2
    JOIN_RATIO=1.0

    echo "=========================================================="
    echo "Running $ALGORITHM for Scenario: $scenario (Clients: $nc)"
    echo "Config: GR=$GR, LR=$LR, LBS=$LBS, LS=$LS, JR=$JOIN_RATIO"
    echo "=========================================================="

    python -u main.py \
        -data "$DATASET" \
        -m "$MODEL" \
        -algo "$ALGORITHM" \
        -gr "$GR" \
        -lr "$LR" \
        -lbs "$LBS" \
        -ls "$LS" \
        -nc "$nc" \
        -jr "$JOIN_RATIO" \
        -go "$GOAL" \
        -dev "$DEVICE" \
        -did "$DEVICE_ID"
done

echo "All ${ALGORITHM} baseline experiments completed."
