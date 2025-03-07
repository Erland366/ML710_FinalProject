import os

import torch
import picotron.process_group_manager as pgm

from transformers import set_seed, LlamaForCausalLM, AutoConfig
from torch import distributed as dist
from dotenv import load_dotenv
from picotron.process_group_manager import setup_process_group_manager
from picotron.data import MicroBatchDataLoader
from picotron.utils import download_model, to_readable_format
from picotron.data_parallel.data_parallel import DataParallelNaive, DataParallelBucket
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from picotron.pipeline_parallel.pipeline_parallel import PipelineParallel
from pydantic_config import parse_argv
from pydantic import validate_call
from safetensors.torch import load_file

from ml710.config import TrainConfig, DataConfig, ParallelConfig, ModelConfig
from ml710.utils import create_logger
from ml710.checkpoint import init_model_with_dematerialized_weights, init_model_with_materialized_weights

load_dotenv()

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
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

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

    if not train_config.pretrain and pgm.process_group_manager.global_rank == 0:
        download_model(model_config.name, os.environ["HF_TOKEN"])

    dist.barrier()

    tokens_per_step = data_loader.global_batch_size + train_config.max_seq_length


    if pgm.process_group_manager.global_rank == 0:
        logger.info("Creating model config")
        config = AutoConfig.from_pretrained(model_config.name)
        objects = [config]
    else:
        objects = [None]

    dist.broadcast_object_list(objects, src=0, device=device)

    config = objects[0]

    logger.info("Broadcasting config to all ranks")

    dist.barrier()

    if global_rank == 0 and train_config.use_wandb:
        import wandb

        config_dict = {}
        config_dict.update(train_config.dict())
        config_dict.update(data_config.dict())
        config_dict.update(parallel_config.dict())
        config_dict.update(model_config.dict())
        config_dict.update(config.to_dict())

        wandb.init(
            project="ml710",
            name=f"{train_config.run_name}-{to_readable_format(tokens_per_step)}-{pgm.process_group_manager}",
            config=config_dict
        )

    logger.debug("")

    # BELOW LOADING IS STILL LOADING FULL MODEL ON EACH GPU
    with init_model_with_dematerialized_weights():
        model = LlamaForCausalLM(config)

        # Still buggy, need to fix this!
        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, config)

    model = init_model_with_materialized_weights(model, config, save_dir=f"./hf_model_safetensors/")

    model.to(dtype).to(device)

    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)

    if global_rank == 0 and train_config.use_wandb:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == '__main__':
    main(**parse_argv())