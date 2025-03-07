import os

import torch
import picotron.process_group_manager as pgm

from torch import distributed as dist
from pydantic_config import parse_argv
from pydantic import validate_call
from picotron.process_group_manager import setup_process_group_manager

from ml710.config import TrainConfig, DataConfig, ParallelConfig

@validate_call
def main(
    train_config: TrainConfig, 
    data_config: DataConfig, 
    parallel_config: ParallelConfig
):
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "gloo" if torch.cuda.is_available() else "nccl"

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

    print(f"[Rank {global_rank}] Hello, world!")
    print(f"{pgm.process_group_manager.tp_world_size = }")

if __name__ == '__main__':
    main(**parse_argv())