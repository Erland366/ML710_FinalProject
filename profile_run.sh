torchrun --nproc_per_node 2 \
 src/ml710/train.py @ configs/DP/naive/DP_naive_1.yaml \
 --train_config.lr 0.001 \
 --train_config.run_profile True 