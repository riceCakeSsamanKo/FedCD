#!/bin/bash

# Common runtime settings
MODEL="VGG16"
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

        # Method-specific hyperparameters from each paper.
        # Heterogeneity scenarios (dataset split and client count) are kept unchanged.
        GR=100
        LR=0.01
        LBS=10
        LS=1
        JOIN_RATIO=1.0
        EXTRA_ARGS=()

        case "$algo" in
            Local)
                # Local training baseline: use the same CIFAR-10 optimizer setting as FedAvg.
                # (FedAvg paper setting on CIFAR-10: E=5, B=50, lr decay 0.99)
                LR=0.15
                LBS=50
                LS=5
                EXTRA_ARGS+=(-ld True -ldg 0.99)
                ;;
            FedAvg)
                # McMahan et al. (AISTATS'17, CIFAR-10): C=0.1, E=5, B=50.
                # FedAvg learning-rate sweep includes {0.05, 0.15, 0.25}; use 0.15.
                LR=0.15
                LBS=50
                LS=5
                JOIN_RATIO=0.1
                EXTRA_ARGS+=(-ld True -ldg 0.99)
                ;;
            FedProx)
                # FedProx (MLSys'20): SGD local solver, batch size 10, tune mu in {0.001,0.01,0.1,1}.
                # For CV-style setting, use mu=1.0 and keep 10 participating clients per round.
                LR=0.15
                LBS=10
                LS=20
                EXTRA_ARGS+=(-mu 1.0)
                if (( nc > 10 )); then
                    JOIN_RATIO=$(awk "BEGIN { printf \"%.6f\", 10 / $nc }")
                fi
                ;;
            FedKD)
                # FedKD (Nature Communications'22) parameters used by this implementation:
                # mentor lr, mentee lr, and dynamic SVD thresholds.
                LR=0.005
                LBS=10
                LS=1
                EXTRA_ARGS+=(-mlr 0.005 -Ts 0.95 -Te 0.98)
                ;;
            FedAS)
                # FedAS paper setting with global rounds aligned to this benchmark.
                GR=100
                LR=0.005
                LBS=16
                LS=5
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
