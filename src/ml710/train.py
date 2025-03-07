import os

import torch
import picotron.process_group_manager as pgm

from transformers import set_seed
from torch import distributed as dist
from picotron.process_group_manager import setup_process_group_manager
from picotron.data import MicroBatchDataLoader
from picotron.utils import download_model
from pydantic_config import parse_argv
from pydantic import validate_call

from ml710.config import TrainConfig, DataConfig, ParallelConfig, ModelConfig
from ml710.utils import create_logger



@validate_call
def main(
    train_config: TrainConfig, 
    data_config: DataConfig, 
    parallel_config: ParallelConfig,
    model_config: ModelConfig
):
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "gloo" if not torch.cuda.is_available() else "nccl"

    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        rank=global_rank,
        world_size=world_size,
        backend=backend,
        init_method="env://",
    )

    setup_process_group_manager(
        tp_size=parallel_config.tp_size,
        pp_size=parallel_config.pp_size,
        dp_size=parallel_config.dp_size,
        cp_size=1
    )

    logger = create_logger(pgm, name="ml710")

    logger.info(f"{pgm.process_group_manager.tp_world_size = }")

    set_seed(train_config.seed)

    data_loader = MicroBatchDataLoader(
        micro_batch_size=train_config.per_device_train_batch_size,
        seq_length=train_config.max_seq_length,
        dataset_name=data_config.path,
        tokenizer_name=model_config.name,
        grad_acc_steps=train_config.gradient_accumulation_steps,
        device=device,
        num_workers=data_config.num_workers,
        num_proc=data_config.num_proc,
        num_samples=train_config.num_samples,
        subset_name=data_config.subset_name,
        split=data_config.split
    )

    if global_rank == 0 and train_config.use_wandb:
        import wandb

        wandb.init(
            project="ml710",
            name=f"{model_config.name}-{data_config.path}",
        )

    dist.barrier()


if __name__ == '__main__':
    main(**parse_argv())