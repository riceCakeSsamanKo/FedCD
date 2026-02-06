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


# List of client counts to test
CLUSTERS_COUNTS=(5 10 20)
CLIENT_COUNTS=(20 50)

echo "============================================================"
echo "Starting Experiment Suite for FedCD"
echo "Client Counts to Test: ${CLIENT_COUNTS[*]}"
echo "============================================================"


for NUM_CLUSTERS in "${CLUSTERS_COUNTS[@]}"
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
        echo "Running Experiments for NUM_CLIENTS = $NUM_CLIENTS"
        echo "NUM_CLUSTERS = $NUM_CLUSTERS"
        echo "Adjusted cluster_sample_size = $CLUSTER_SAMPLE_SIZE"
        echo "############################################################"

        # ------------------------------------------------------------------
        # Experiment 1: Pathological Non-IID (pat) - Balanced
        # ------------------------------------------------------------------
        echo ""
        echo ">>> [Exp 1/2] Setting up Pathological Non-IID (pat) - Balanced | Clients: $NUM_CLIENTS"
        echo "Cleaning up old dataset partition..."
        rm -f dataset/$DATASET/config.json
        rm -rf dataset/$DATASET/train
        rm -rf dataset/$DATASET/test

        echo "Generating Dataset..."
        # [Fix] Change directory to dataset/ to ensure Cifar10 folder is created inside dataset/
        (cd dataset && python generate_Cifar10.py noniid balance pat $NUM_CLIENTS) || echo "Warning: Dataset generation (pat) failed!"

        echo "Running Training (pat)..."
        START_TIME=$SECONDS
        python system/main.py \
            -data $DATASET \
            -algo $ALGO \
            --gm_model VGG16 \
            --pm_model VGG8 \
            -gr $GLOBAL_ROUNDS \
            -nc $NUM_CLIENTS \
            --num_clusters $NUM_CLUSTERS \
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
            --local_epochs 1 || echo "Warning: Training (pat) failed for $NUM_CLIENTS clients. Skipping..."
        ELAPSED_TIME=$(($SECONDS - $START_TIME))

        # Copy dataset config to the latest log directory
        LATEST_LOG_DIR=$(ls -td logs/exp_* | head -n 1)
        if [ -d "$LATEST_LOG_DIR" ]; then
            cp "dataset/$DATASET/config.json" "$LATEST_LOG_DIR/dataset_config_pat_NUM_CLUSTERS_${NUM_CLUSTERS}_NUM_CLIENTS_${NUM_CLIENTS}.json"
            echo "Pathological (pat) execution time: ${ELAPSED_TIME}s" >> "$LATEST_LOG_DIR/time.txt"
            echo "[Shell] Copied dataset config to $LATEST_LOG_DIR"
        fi

        echo ">>> Exp 1 (pat) Finished."
        sleep 5

        # ------------------------------------------------------------------
        # Experiment 2: Dirichlet Non-IID (dir) - Unbalanced
        # ------------------------------------------------------------------
        echo ""
        echo ">>> [Exp 2/2] Setting up Dirichlet Non-IID (dir) - Unbalanced | Clients: $NUM_CLIENTS"
        echo "Cleaning up old dataset partition..."
        rm -f dataset/$DATASET/config.json
        rm -rf dataset/$DATASET/train
        rm -rf dataset/$DATASET/test

        echo "Generating Dataset..."
        # [Fix] Change directory to dataset/ to ensure Cifar10 folder is created inside dataset/
        (cd dataset && python generate_Cifar10.py noniid - dir $NUM_CLIENTS) || echo "지금 Warning: Dataset generation (dir) failed!"

        echo "Running Training (dir)..."
        START_TIME=$SECONDS
        python system/main.py \
            -data $DATASET \
            -algo $ALGO \
            --gm_model VGG16 \
            --pm_model VGG8 \
            -gr $GLOBAL_ROUNDS \
            -nc $NUM_CLIENTS \
            --num_clusters $NUM_CLUSTERS \
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
            --local_epochs 1 || echo "Warning: Training (dir) failed for $NUM_CLIENTS clients. Skipping..."
        ELAPSED_TIME=$(($SECONDS - $START_TIME))

        # Copy dataset config to the latest log directory
        LATEST_LOG_DIR=$(ls -td logs/exp_* | head -n 1)
        if [ -d "$LATEST_LOG_DIR" ]; then
            cp "dataset/$DATASET/config.json" "$LATEST_LOG_DIR/dataset_config_dir_NUM_CLUSTERS_${NUM_CLUSTERS}_NUM_CLIENTS_${NUM_CLIENTS}.json"
            echo "Dirichlet (dir) execution time: ${ELAPSED_TIME}s" >> "$LATEST_LOG_DIR/time.txt"
            echo "[Shell] Copied dataset config to $LATEST_LOG_DIR"
        fi

        echo ">>> Exp 2 (dir) Finished."
        sleep 5
    done

    echo "============================================================"
    echo "All Experiments Completed for clients: ${CLIENT_COUNTS[*]}"
done