#!/bin/bash

# Continue on error (skip failed experiments)
set +e

# Settings
GPU_DEVICE="cuda"
GLOBAL_ROUNDS=100
ALGO="FedCD"
DATASET="Cifar10"
TOTAL_DATA=50000
AVOID_OOM=True

# List of Dirichlet alpha values to test
ALPHAS=(0.1 0.5 1.0) # (0.1 0.5 1.0)
# List of distance thresholds for Agglomerative Clustering
THRESHOLDS=(0.1)
CLIENT_COUNTS=(20 50)

echo "============================================================"
echo "Starting Experiment Suite for FedCD (Adaptive Threshold - ACT)"
echo "Tested Alphas: ${ALPHAS[*]}"
echo "Initial Thresholds to Test: ${THRESHOLDS[*]}"
echo "============================================================"

    for THRESHOLD in "${THRESHOLDS[@]}"
    do
        for NUM_CLIENTS in "${CLIENT_COUNTS[@]}"
        do
            # Calculate safe cluster_sample_size
            AVG_DATA_PER_CLIENT=$((TOTAL_DATA / NUM_CLIENTS))
            if [ "$AVG_DATA_PER_CLIENT" -lt 512 ]; then
                CLUSTER_SAMPLE_SIZE=$AVG_DATA_PER_CLIENT
            else
                CLUSTER_SAMPLE_SIZE=512
            fi
            
            echo ""
            echo "############################################################"
            echo "NUM_CLIENTS = $NUM_CLIENTS"
            echo "CLUSTER_THRESHOLD = $THRESHOLD"
            echo "Adjusted cluster_sample_size = $CLUSTER_SAMPLE_SIZE"
            echo "############################################################"

            # ------------------------------------------------------------------
            # Experiment 1: Pathological Non-IID (pat) - Balanced
            # ------------------------------------------------------------------
            DATASET_NAME="Cifar10_pat_nc${NUM_CLIENTS}"
            echo ""
            echo ">>> [Exp 1/2] Using Pathological (pat) | Clients: $NUM_CLIENTS"
            
            echo "Running Training (pat)..."
            START_TIME=$SECONDS
            python system/main.py \
                -data $DATASET_NAME \
                -algo $ALGO \
                --gm_model VGG16 \
                --pm_model VGG8 \
                -gr $GLOBAL_ROUNDS \
                -nc $NUM_CLIENTS \
                --cluster_threshold $THRESHOLD \
                --adaptive_threshold True \
                --threshold_step 0.05 \
                --threshold_decay 0.9 \
                --act_window_size 5 \
                --cluster_period 2 \
                --pm_period 1 \
                --global_period 4 \
                --cluster_sample_size $CLUSTER_SAMPLE_SIZE \
                -dev $GPU_DEVICE \
                -nw 0 \
                --pin_memory True \
                --prefetch_factor 2 \
                --amp True \
                --tf32 True \
                --gpu_batch_mult 32 \
                --gpu_batch_max 0 \
                --log_usage True \
                --avoid_oom $AVOID_OOM \
                --local_epochs 1 \
                --proxy_dataset TinyImagenet --proxy_samples 2000 || echo "Warning: Training (pat) failed for $NUM_CLIENTS clients. Skipping..."
            ELAPSED_TIME=$(($SECONDS - $START_TIME))

            # Copy dataset config to the latest log directory from fl_data
            LATEST_LOG_DIR=$(find logs -type d -name "time_*" | xargs ls -td | head -n 1)
            if [ -d "$LATEST_LOG_DIR" ]; then
                cp "../fl_data/$DATASET_NAME/config.json" "$LATEST_LOG_DIR/dataset_config_pat_THRESHOLD_${THRESHOLD}_NUM_CLIENTS_${NUM_CLIENTS}.json"
                echo "Pathological (pat) execution time: ${ELAPSED_TIME}s" >> "$LATEST_LOG_DIR/time.txt"
                echo "[Shell] Copied dataset config from fl_data to $LATEST_LOG_DIR"
            fi

            echo ">>> Exp 1 (pat) Finished."
            sleep 5

            for ALPHA in "${ALPHAS[@]}"
            do
                echo ""
                echo "------------------------------------------------------------"
                echo "Running Dirichlet for ALPHA = $ALPHA"
                echo "------------------------------------------------------------"

                # ------------------------------------------------------------------
                # Experiment 2: Dirichlet Non-IID (dir) - Unbalanced
                # ------------------------------------------------------------------
                DATASET_NAME="Cifar10_dir${ALPHA}_nc${NUM_CLIENTS}"
                echo ""
                echo ">>> [Exp 2/2] Using Dirichlet (dir) | Alpha: $ALPHA | Clients: $NUM_CLIENTS"

                echo "Running Training (dir)..."
                START_TIME=$SECONDS
                python system/main.py \
                    -data $DATASET_NAME \
                    -algo $ALGO \
                    --gm_model VGG16 \
                    --pm_model VGG8 \
                    -gr $GLOBAL_ROUNDS \
                    -nc $NUM_CLIENTS \
                    --cluster_threshold $THRESHOLD \
                    --adaptive_threshold True \
                    --threshold_step 0.05 \
                    --threshold_decay 0.9 \
                    --act_window_size 5 \
                    --cluster_period 2 \
                    --pm_period 1 \
                    --global_period 4 \
                    --cluster_sample_size $CLUSTER_SAMPLE_SIZE \
                    -dev $GPU_DEVICE \
                    -nw 0 \
                    --pin_memory True \
                    --prefetch_factor 2 \
                    --amp True \
                    --tf32 True \
                    --gpu_batch_mult 32 \
                    --gpu_batch_max 0 \
                    --log_usage True \
                    --avoid_oom $AVOID_OOM \
                    --local_epochs 1 \
                    --proxy_dataset TinyImagenet --proxy_samples 2000|| echo "Warning: Training (dir) failed for $NUM_CLIENTS clients. Skipping..."
                ELAPSED_TIME=$(($SECONDS - $START_TIME))

                # Copy dataset config to the latest log directory from fl_data
                LATEST_LOG_DIR=$(find logs -type d -name "time_*" | xargs ls -td | head -n 1)
                if [ -d "$LATEST_LOG_DIR" ]; then
                    cp "../fl_data/$DATASET_NAME/config.json" "$LATEST_LOG_DIR/dataset_config_dir_ALPHA_${ALPHA}_THRESHOLD_${THRESHOLD}_NUM_CLIENTS_${NUM_CLIENTS}.json"
                    echo "Dirichlet (dir) execution time: ${ELAPSED_TIME}s" >> "$LATEST_LOG_DIR/time.txt"
                    echo "[Shell] Copied dataset config from fl_data to $LATEST_LOG_DIR"
                fi
            done
            echo ">>> Exp 2 (dir) Finished for NUM_CLIENTS=$NUM_CLIENTS"
            sleep 5
        done
    done

echo "============================================================"
echo "All experiments completed."
echo "============================================================"
