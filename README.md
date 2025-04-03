# Plan
- [x] Implement [MicroLLama](https://github.com/keeeeenw/MicroLlama)
- [x] Implement [Picotron](https://github.com/huggingface/picotron) on top of MicroLlama
- [ ] Implement AMP
- Decide on the dataset for training

# Target
We need to make sure that
- FSDP should be slower than the rest of DP
- Bucketing is faster than Naive
- FSDP requires less memory than the others
- Same loss across any implementations

# Parallel Strategies
- [x] Data Parallelism
    - [x] Naive
    - [x] Wait-free backpropagation
    - [x] Bucketing
    - [x] FSDP
- [x] Model Parallelism
- [x] Pipeline Parallelism
<!-- - [ ] Data + Model Parallelism (Need at least 4 GPUs) -->

# How to Install
```
conda create -n ml710_finalproject python=3.11 cudatoolkit-dev -c conda-forge -y
conda activate ml710_finalproject
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Last, install this repo by :
```
pip install -e .
```

# Run the Program
Example command to run the program
```
python src/ml710/train.py @ configs/base.yaml
```

or if using torchrun
```
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 2 src/ml710/train.py @ configs/base.yaml
```

Modify your config on CLI by the power of [pydantic_config](https://github.com/samsja/pydantic_config)
```
WANDB_MODE=offline CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node 1 src/ml710/train.py @ configs/base.yaml --parallel_config.dp_size=1
```

If you want to debug, you can use `debugpy-run` like this 

```
CUDA_DEVICE_MAX_CONNECTIONS=1 debugpy-run -m torch.distributed.run -- --nproc_per_node 2 src/ml710/train.py @ configs/base.yaml
```

Remove `CUDA_DEVICE_MAX_CONNECTIONS=1` on FSDP
