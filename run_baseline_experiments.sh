#!/bin/bash

bucket_sizes=(1 8 16 32 64)

echo "--- Running Parallelism Experiment: Baseline"

for n in "${bucket_sizes[@]}"; do
    echo "--- Running with bucket size: ${n} ---"
        config_file="configs/baseline/baseline_${n}.yaml"
        if [ -f "$config_file" ]; then
            echo "Running with ${config_file}"
            CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
                --nproc_per_node 1 \
                src/ml710/train.py @"$config_file" --train_config.lr 0.001
            echo "Finished running ${config_file}"
        else
            echo "Warning: Config file not found, skipping: ${config_file}"
        fi
        echo
done

echo "--- Completed all sizes for Baseline ---"
echo

echo "All Baseline experiments completed."