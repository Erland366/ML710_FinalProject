#!/bin/bash

bucket_sizes=(1 8 16 32 64 72)
sequence_length=(512 4096)
parallelism_types=(sync async)
time_or_tokens=("time" "tokens")

for type in "${parallelism_types[@]}"; do
    echo "--- Running Parallelism Experiment: TP_${type} ---"

    for n in "${bucket_sizes[@]}"; do
        echo "--- Running with bucket size: ${n} ---"

        for seq_len in "${sequence_length[@]}"; do
            echo "--- Running with sequence length: ${seq_len} ---"

            for time_or_token in "${time_or_tokens[@]}"; do
                echo "--- Running with time_or_token: ${time_or_token} ---"
                config_file="configs/TP/TP_${type}/TP${type}_${n}_${seq_len}_${time_or_token}.yaml"

                if [ -f "$config_file" ]; then
                    echo "Running with ${config_file}"
                    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
                        --nproc_per_node 2 \
                        src/ml710/train.py @"$config_file" --train_config.lr 0.001
                    echo "Finished running ${config_file}"
                else
                    echo "Warning: Config file not found, skipping: ${config_file}"
                fi
                echo
            done
        done
    done

    echo "--- Completed all sizes for TP_${type} ---"
    echo
done

echo "All TP experiments completed."