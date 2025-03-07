import inspect
import os
import time

import torch
import torch.nn.functional as F
import picotron.process_group_manager as pgm

from transformers import set_seed, LlamaForCausalLM, AutoConfig
from torch import distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from dotenv import load_dotenv
from picotron.process_group_manager import setup_process_group_manager
from picotron.data import MicroBatchDataLoader
from picotron.utils import average_loss_across_dp_cp_ranks, set_all_seed, print, to_readable_format, get_mfu, get_num_params, download_model
from picotron.data_parallel.data_parallel import DataParallelNaive, DataParallelBucket
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from picotron.pipeline_parallel.pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from pydantic_config import parse_argv
from pydantic import validate_call
from safetensors.torch import load_file

from ml710.config import TrainConfig, DataConfig, ParallelConfig, ModelConfig
from ml710.utils import create_logger
from ml710.checkpoint import init_model_with_dematerialized_weights, init_model_with_materialized_weights, CheckpointManager
from ml710.metrics import GoodputMetrics

load_dotenv()

def train_step(model, data_loader, device):
    acc_loss = 0.0
    
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    for i in range(data_loader.grad_acc_steps):
        # get the next batch
        batch = next(data_loader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # disable gradient synchronization for all but the last micro-batch
        if requires_grad_sync:
            model.require_backward_grad_sync = (i == data_loader.grad_acc_steps - 1)

        outputs = model(input_ids=input_ids)

        # compute the loss
        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)

        outputs = outputs.logits.view(seq_len*batch_size, -1)
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

        # Change attention implementation here
        config._attn_implementation = train_config.attn_implementation

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

    with init_model_with_dematerialized_weights():
        model = LlamaForCausalLM(config)

        # Still buggy, need to fix this!
        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model)

        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, config)


    model = init_model_with_materialized_weights(model, config, save_dir=f"./hf_model_safetensors/")

    model.to(dtype).to(device)

    # Compile
    if train_config.use_compile:
        model = torch.compile(model)

    if pgm.process_group_manager.dp_world_size > 1:
        if parallel_config.dp_engine == "naive":
            model = DataParallelNaive(model)
        elif parallel_config.dp_engine == "bucket":
            model = DataParallelBucket(model)
        elif parallel_config.dp_engine == "ddp":
            model = DDP(model, device_ids=[local_rank])
        else:
            raise ValueError(f"Invalid data parallel engine: {parallel_config.dp_engine}")

    model.train()
    num_params = get_num_params(model)
    if global_rank == 0:
        logger.info(f"Number of parameters: {to_readable_format(num_params)}")
    
    tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, config.hidden_size)
    
    extra_args = dict()
    if train_config.use_fused_adam:
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

    optimizer = AdamW(model.parameters(), lr=train_config.lr, **extra_args)
    
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
        trained_tokens += tokens_per_step
        step += 1
        
        if hasattr(model, 'reset'):
            model.reset()

        step_duration = time.time() - step_start_time
        tokens_per_second = tokens_per_step / step_duration
        tokens_per_second_per_gpu = tokens_per_second / world_size
        mfu = get_mfu(tokens_per_second_per_gpu, num_params, config)
        
        if global_rank == 0:
            goodput_log = goodput_metrics.metrics(step_duration, loss)
            logger.info(
                f"[rank {pgm.process_group_manager.global_rank}] "
                f"Step: {step:<5d} | "
                f"Loss: {loss:6.4f} | "
                f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                f"Tok/s: {to_readable_format(tokens_per_second):>7s} | "
                f"Tok/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                f"Tok: {to_readable_format(trained_tokens):>7s}{('/' + to_readable_format(train_config.max_tokens)) if train_config.max_tokens else ''} | "
                f"MFU: {mfu:5.2f}% | "
                f"Memory usage: {torch.cuda.memory_reserved() / 1e9:6.2f}GB | "
                f"T: {goodput_log['throughput']:6.4f} | "
                f"SE: {goodput_log['statistical_efficiency']:6.4f} | "
                f"G: {goodput_log['goodput']:6.4f} | "
            )
        
            if train_config.use_wandb:
                wandb_log = {
                    "loss": loss,
                    "tokens_per_step": tokens_per_step,
                    "tokens_per_second": tokens_per_step / step_duration,
                    "mfu": mfu,
                    "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                    "memory_usage": torch.cuda.memory_reserved() / 1e9,
                    "trained_tokens": trained_tokens,
                    "goodput": goodput_log["goodput"],
                    "throughput": goodput_log["throughput"],
                    "statistical_efficiency": goodput_log["statistical_efficiency"]
                }
                wandb.log(wandb_log)
        
        if step % train_config.save_frequency == 0:
            checkpoint_manager.save_checkpoint(model, optimizer, step, trained_tokens, train_config.checkpoint_path+f"/{step}")
        
        if step >= train_config.total_train_steps:
            break

    if global_rank == 0 and train_config.use_wandb:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == '__main__':
    main(**parse_argv())