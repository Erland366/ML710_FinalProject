#!/bin/bash

bucket_sizes=(1 8 16 32 64)
parallelism_types=(naive wait_free fsdp bucket)

for type in "${parallelism_types[@]}"; do
    echo "--- Running Parallelism Experiment: DP_${type} ---"

    for n in "${bucket_sizes[@]}"; do
        config_file="configs/DP/${type}/DP_${type}_${n}.yaml"

        if [ -f "$config_file" ]; then
            echo "Running with ${config_file}"
            CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
                --nproc_per_node 2 \
                src/ml710/train.py @ "$config_file" --train_config.lr 0.001
            echo "Finished running ${config_file}"
        else
            echo "Warning: Config file not found, skipping: ${config_file}"
        fi
        echo 
    done

    echo "--- Completed all sizes for DP_${type} ---"
    echo 
done

echo "All DP experiments completed."
