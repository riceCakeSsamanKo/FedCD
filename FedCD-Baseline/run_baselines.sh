#!/bin/bash

# Common runtime settings
MODEL="VGG8"
DEVICE="cuda"
DEVICE_ID="0"

# Algorithms to test
algorithms=("Local" "FedProx" "FedKD" "FedAS" "FedAvg") 

# Scenarios available in fl_data
scenarios=(
    "pat_nc20" "dir0.1_nc20" "dir0.5_nc20" "dir1.0_nc20"
    "pat_nc50" "dir0.1_nc50" "dir0.5_nc50" "dir1.0_nc50"
)

# Go to system directory
cd system || exit 1

for algo in "${algorithms[@]}"; do
    for scenario in "${scenarios[@]}"; do
        DATASET="Cifar10_$scenario"
        
        if [[ $scenario =~ nc([0-9]+) ]]; then
            nc=${BASH_REMATCH[1]}
        else
            nc=20
        fi
        
        GOAL="${algo}_${scenario}"

        # Common hyperparameters aligned with FedCD defaults.
        GR=100
        LR=0.005
        LBS=128
        LS=2
        JOIN_RATIO=1.0
        EXTRA_ARGS=()

        case "$algo" in
            FedProx)
                # Keep method-specific proximal regularization.
                EXTRA_ARGS+=(-mu 1.0)
                ;;
            FedKD)
                # FedKD-specific defaults in this codebase.
                EXTRA_ARGS+=(-mlr 0.005 -Ts 0.95 -Te 0.98)
                ;;
        esac

        echo "=========================================================="
        echo "Running $algo for Scenario: $scenario (Clients: $nc)"
        echo "Config: GR=$GR, LR=$LR, LBS=$LBS, LS=$LS, JR=$JOIN_RATIO"
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
            -jr "$JOIN_RATIO" \
            -go "$GOAL" \
            -dev "$DEVICE" \
            -did "$DEVICE_ID" \
            "${EXTRA_ARGS[@]}"
        
        # No need to move usage.csv anymore
    done
done

echo "All baseline experiments completed."
