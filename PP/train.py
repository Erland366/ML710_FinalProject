import inspect
import os
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import picotron.process_group_manager as pgm

from transformers import set_seed, LlamaForCausalLM, AutoConfig
from torch import distributed as dist
from torch.optim import AdamW
from dotenv import load_dotenv
from picotron.process_group_manager import setup_process_group_manager
from picotron.data import MicroBatchDataLoader
from picotron.utils import average_loss_across_dp_cp_ranks, set_all_seed, print, to_readable_format, get_mfu, get_num_params, download_model
from picotron.pipeline_parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from pydantic_config import parse_argv
from pydantic import validate_call
from safetensors.torch import load_file

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(os.path.join(project_root, "src"))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import ml710.path_setup

from ml710.config import TrainConfig, DataConfig, ParallelConfig, ModelConfig
from ml710.utils import create_logger
from ml710.checkpoint import init_model_with_dematerialized_weights, init_model_with_materialized_weights, CheckpointManager
from ml710.metrics import GoodputMetrics

load_dotenv()

def train_step(model, data_loader, device):
    acc_loss = 0.0
    
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    for i in range(data_loader.grad_acc_steps):
        batch = next(data_loader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        if requires_grad_sync:
            model.require_backward_grad_sync = (i == data_loader.grad_acc_steps - 1)

        outputs = model(input_ids=input_ids)

        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)

        outputs = outputs.logits.view(seq_len * batch_size, -1)
        loss = F.cross_entropy(outputs, target_ids, reduction='mean') / data_loader.grad_acc_steps
        
        loss.backward()
        acc_loss += loss.item()

    return acc_loss

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

    # Set up process groups for Pipeline Parallelism
    setup_process_group_manager(
        tp_size=parallel_config.tp_size,
        pp_size=4,  # Set PP to 4 GPUs
        dp_size=1,  # Disable DP
        cp_size=1
    )

    logger = create_logger(pgm, name="ml710")
    set_seed(train_config.seed)

    # Initialize Data Loader
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

    # Download the model if needed
    if not train_config.pretrain and pgm.process_group_manager.global_rank == 0:
        download_model(model_config.name, os.environ["HF_TOKEN"])

    dist.barrier()

    if pgm.process_group_manager.global_rank == 0:
        logger.info("Creating model config")
        config = AutoConfig.from_pretrained(model_config.name)
        config._attn_implementation = train_config.attn_implementation
        objects = [config]
    else:
        objects = [None]

    dist.broadcast_object_list(objects, src=0, device=device)
    config = objects[0]

    logger.info("Broadcasting config to all ranks")
    dist.barrier()

    # Initialize Model
    with init_model_with_dematerialized_weights():
        model = LlamaForCausalLM(config)

        # Apply Pipeline Parallelism
        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, config)

    model = init_model_with_materialized_weights(model, config, save_dir=f"./hf_model_safetensors/")
    model.to(dtype).to(device)

    if train_config.use_compile:
        model = torch.compile(model)

    model.train()
    num_params = get_num_params(model)
    if global_rank == 0:
        logger.info(f"Number of parameters: {to_readable_format(num_params)}")

    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, config.hidden_size)

    optimizer = AdamW(model.parameters(), lr=train_config.lr)

    checkpoint_manager = CheckpointManager()

    trained_tokens, step = 0, 0
    if train_config.load_path:
        step, trained_tokens = checkpoint_manager.load_checkpoint(model, optimizer, train_config.load_path)

    dist.barrier()

    goodput_metrics = GoodputMetrics(window_size=1, mini_batch_size=train_config.per_device_train_batch_size * pgm.process_group_manager.dp_world_size)
    goodput_metrics.reset_time()

    while train_config.max_tokens is None or trained_tokens < train_config.max_tokens:
        step_start_time = time.time()
        optimizer.zero_grad()

        if pgm.process_group_manager.pp_world_size > 1:
            if parallel_config.pp_engine == "afab":
                loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype)
            elif parallel_config.pp_engine == "1f1b":
                loss = train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype)
            else:
                raise ValueError(f"Invalid pipeline parallel engine: {parallel_config.pp_engine}")
        else:
            loss = train_step(model, data_loader, device)
            
        loss = average_loss_across_dp_cp_ranks(loss, device)

        optimizer.step()
        trained_tokens += data_loader.global_batch_size + train_config.max_seq_length
        step += 1

        if hasattr(model, 'reset'):
            model.reset()

        step_duration = time.time() - step_start_time
        tokens_per_second = trained_tokens / step_duration

        if global_rank == 0:
            logger.info(f"Step: {step} | Loss: {loss:.4f} | Tokens/sec: {to_readable_format(tokens_per_second)}")

        if step >= train_config.total_train_steps:
            break

    dist.destroy_process_group()

if __name__ == '__main__':
    main(**parse_argv())
