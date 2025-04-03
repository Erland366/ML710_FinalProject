#!/bin/bash

# Values to substitute for 'n'
bucket_sizes=(1 8 16 32 64)

# Loop through each bucket size
for n in "${bucket_sizes[@]}"; do
    echo "Running with DP_bucket_${n}.yaml"
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun \
        --nproc_per_node 2 \
        src/ml710/train.py @ "configs/DP/bucket/DP_bucket_${n}.yaml"
done

echo "All bucket sizes completed"
