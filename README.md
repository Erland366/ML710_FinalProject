# Plan
- Implement [MicroLLama](https://github.com/keeeeenw/MicroLlama)
- Implement [Picotron](https://github.com/huggingface/picotron) on top of MicroLlama
- Decide on the dataset for training


# Parallel Strategies
- [ ] Data Parallelism
- [ ] Model Parallelism
- [ ] Pipeline Parallelism
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

Remove `CUDA_DEVICE_MAX_CONNECTIONS=1` on FSDP