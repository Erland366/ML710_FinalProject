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
conda create -n ml710_finalproject python=3.11 -y
conda activate ml710_finalproject
pip install -r requirements.txt --no-dependencies
```

If you have problem to install flash-attn:
```
conda install cudatoolkit-dev -y
CUDA_HOME=~/miniconda3/envs/ml710_finalproject pip install flash-attn
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
torchrun --nproc_per_node 2 src/ml710/train.py @ configs/base.yaml
```