#!/bin/bash

# Experiment settings
MODEL="VGG16"
GR=100
LR=0.01
LBS=10
LS=1
DEVICE="cuda"
DEVICE_ID="0"

# Algorithms to test
algorithms=("FedAvg" "FedProx" "FedAS" "Local" "FedKD")

# Scenarios available in fl_data
scenarios=(
    "pat_nc50" "dir0.1_nc50" "dir0.5_nc50" "dir1.0_nc50"
)

# Go to system directory
cd system

for algo in "${algorithms[@]}"; do
    for scenario in "${scenarios[@]}"; do
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
        
        # Logs are now automatically handled by main.py and serverbase.py in FedCD style
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
        
        # No need to move usage.csv anymore
    done
done

echo "All baseline experiments completed."
