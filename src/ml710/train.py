"""
model = LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 1024)
    (layers): ModuleList(
      (0-11): 12 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (k_proj): Linear(in_features=1024, out_features=256, bias=False)
          (v_proj): Linear(in_features=1024, out_features=256, bias=False)
          (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=1024, out_features=5632, bias=False)
          (up_proj): Linear(in_features=1024, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=1024, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((1024,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((1024,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=32000, bias=False)
)
"""

import inspect
import os
import time
import wandb

import torch
import torch.nn.functional as F
import picotron.process_group_manager as pgm
import pandas as pd

from functools import partial

from transformers import set_seed, LlamaForCausalLM, AutoConfig
from torch import distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from dotenv import load_dotenv
from picotron.process_group_manager import setup_process_group_manager
from picotron.data import MicroBatchDataLoader
from picotron.utils import average_loss_across_dp_cp_ranks, set_all_seed, print, to_readable_format, get_mfu, get_num_params, download_model
from picotron.data_parallel.data_parallel import DataParallelNaive as DataParallelWaitFree, DataParallelBucket
from picotron.tensor_parallel.tensor_parallel import apply_tensor_parallel
from pipeline_parallel import train_step_pipeline_1f1b, train_step_pipeline_afab, PipelineParallel
from pydantic_config import parse_argv
from pydantic import validate_call
from contextlib import nullcontext
from safetensors.torch import load_file
from torch.profiler import profile, record_function, ProfilerActivity
from torch._C._profiler import _ExperimentalConfig

from ml710.config import TrainConfig, DataConfig, ParallelConfig, ModelConfig
from ml710.utils import create_logger
from ml710.checkpoint import init_model_with_dematerialized_weights, init_model_with_materialized_weights, CheckpointManager
from ml710.metrics import GoodputMetrics
from ml710.data_parallel import FSDP, DataParallelNaive
from ml710.profiler import trace_handler

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
    assert any([
        train_config.max_tokens, 
        train_config.max_time, 
        train_config.max_steps, 
        train_config.max_loss
    ]), "Please specify one of max_tokens, max_time, max_steps, or max_loss"

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

    # Setup profiler
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    use_default_schedule = not any([
        train_config.run_profile, 
        train_config.wait_steps, 
        train_config.warmup_steps, 
        train_config.active_steps, 
        train_config.num_cycles
    ])

    DEFAULT_PROFILE_ARGS = {
        "wait_steps" : 4,
        "warmup_steps" : 4,
        "active_steps" : 4,
        "num_cycles" : 2
    }

    if use_default_schedule:
        scheduler_args = DEFAULT_PROFILE_ARGS
        if global_rank == 0 and train_config.run_profile:
            logger.info(
                "No schedule found in config, default to {}".format(
                    ", ".join(f"{k}={v}" for k, v in scheduler_args.items())
                )
            )

    else:
        scheduler_args = {
            "wait_steps" : train_config.wait_steps,
            "warmup_steps" : train_config.warmup_steps,
            "active_steps" : train_config.active_steps,
            "num_cycles" : train_config.num_cycles
        }
        missing_args = [k for k, v in scheduler_args.items() if v is None]
        if len(missing_args) > 0:
            for arg in missing_args:
                scheduler_args[arg] = DEFAULT_PROFILE_ARGS[arg]
            if global_rank == 0 and train_config.run_profile:
                logger.info(
                    "Missing schedule arguments, default to {}".format(
                        ", ".join(f"{k}={v}" for k, v in scheduler_args.items())
                    )
                )

    schedule = torch.profiler.schedule(
        wait=scheduler_args["wait_steps"],
        warmup=scheduler_args["warmup_steps"],
        active=scheduler_args["active_steps"],
        repeat=scheduler_args["num_cycles"]
    )

    experimental_config = _ExperimentalConfig(verbose=True) if train_config.with_stack else None

    output_dir = train_config.profile_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callback = partial(trace_handler, output_dir=output_dir)

    prof = torch.profiler.profile(
        activities=activities,
        profile_memory=train_config.profile_memory,
        with_stack=train_config.with_stack,
        record_shapes=train_config.record_shapes,
        with_flops=train_config.with_flops,
        schedule=schedule,
        experimental_config=experimental_config,
        on_trace_ready=callback
    ) if train_config.run_profile else nullcontext()

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

    is_wandb_rank = pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.dp_rank == 0 and pgm.process_group_manager.cp_rank == 0 and pgm.process_group_manager.pp_is_last_stage

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

    if is_wandb_rank:
        # Better run name for logging experiments
        if pgm.process_group_manager.tp_world_size > 1:
            train_config.run_name += f"-tp_{parallel_config.tp_engine.upper()}"
        if pgm.process_group_manager.pp_world_size > 1:
            train_config.run_name += f"-pp_{parallel_config.pp_engine.upper()}"
        if pgm.process_group_manager.dp_world_size > 1:
            train_config.run_name += f"-dp_{parallel_config.dp_engine.upper()}"

        config_dict = {}
        config_dict.update(train_config.dict())
        config_dict.update(data_config.dict())
        config_dict.update(parallel_config.dict())
        config_dict.update(model_config.dict())
        config_dict.update(config.to_dict())

        wandb.init(
            project="ml710_FINALFINALFINALFINALproject",
            name=f"{train_config.run_name}-{to_readable_format(tokens_per_step)}-{pgm.process_group_manager}",
            config=config_dict
        )

    with init_model_with_dematerialized_weights():
        model = LlamaForCausalLM(config)
        
        if pgm.process_group_manager.tp_world_size > 1:
            model = apply_tensor_parallel(model, parallel_config.tp_engine == "sync")

        if pgm.process_group_manager.pp_world_size > 1:
            model = PipelineParallel(model, config)

    model = init_model_with_materialized_weights(model, config, save_dir="./hf_model_safetensors/")

    model.to(dtype).to(device)

    # Compile
    if train_config.use_compile:
        model = torch.compile(model)

    if pgm.process_group_manager.dp_world_size > 1:
        if parallel_config.dp_engine == "wait_free":
            model = DataParallelWaitFree(model)
        elif parallel_config.dp_engine == "naive":
            model = DataParallelNaive(model)
        elif parallel_config.dp_engine == "bucket":
            model = DataParallelBucket(model)
        elif parallel_config.dp_engine == "ddp":
            model = DDP(model, device_ids=[local_rank])
        elif parallel_config.dp_engine == "fsdp":
            model = FSDP(model, zero_stage=parallel_config.zero_stage).model
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

    goodput_metrics = GoodputMetrics(
        window_size=1, 
        grad_acc_steps=train_config.gradient_accumulation_steps,
        mini_batch_size=train_config.per_device_train_batch_size * pgm.process_group_manager.dp_world_size
    )
    goodput_metrics.reset_time()

    start_time = time.time()
    end_time = time.time()
    loss = float("inf")
    
    csv_path = os.path.join(".", "results")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    csv_path = csv_path + '/' + train_config.results_file +'.csv'

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            "loss", "tokens_per_step", "tokens_per_second", "mfu", 
            "tokens_per_second_per_gpu", "memory_usage", "trained_tokens", 
            "goodput", "throughput", "statistical_efficiency", "avg_goodput", "avg_throughput", "avg_statistical_efficiency", "batch_size",
            "max_seq_length"
        ])

    results = {}

    with prof as p:
        while (train_config.max_tokens is None 
            or trained_tokens < train_config.max_tokens 
            or train_config.max_time is None 
            or end_time - start_time < train_config.max_time
            or train_config.max_steps is None
            or step < train_config.max_steps
            or train_config.max_loss is None
            or loss > train_config.max_loss
        ):
            step_start_time = time.time()
            optimizer.zero_grad()
            
            #YASMEEN: Initialize goodput_log with default values to avoid UnboundLocalError
            goodput_log = {
                "goodput": 0.0,
                "throughput": 0.0,
                "statistical_efficiency": 0.0
            }

            if pgm.process_group_manager.pp_world_size > 1:
                if parallel_config.pp_engine == "afab":
                    loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype)
                elif parallel_config.pp_engine == "1f1b":
                    loss = train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype)
                else:
                    raise ValueError(f"Invalid pipeline parallel engine: {parallel_config.pp_engine}")
            else:
                loss = train_step(model, data_loader, device)

            if train_config.run_profile and global_rank == 0:
                p.step()
                
            loss = average_loss_across_dp_cp_ranks(loss, device)

            if parallel_config.dp_engine == "naive":
                model.synchronize_gradients()
            
            optimizer.step()
            trained_tokens += tokens_per_step
            step += 1
            
            if hasattr(model, 'reset'):
                model.reset()

            step_duration = time.time() - step_start_time
            tokens_per_second = tokens_per_step / step_duration
            tokens_per_second_per_gpu = tokens_per_second / world_size
            mfu = get_mfu(tokens_per_second_per_gpu, num_params, config)

            if is_wandb_rank:
                goodput_log = goodput_metrics.metrics(time.time(), loss)
                experiment_print = None
                if train_config.max_tokens:
                    experiment_print = f"Tok: {to_readable_format(trained_tokens):>7s}{('/' + to_readable_format(train_config.max_tokens)) if train_config.max_tokens else ''} | "
                elif train_config.max_time:
                    experiment_print = f"Time: {to_readable_format(end_time - start_time):>7s}{('/' + to_readable_format(train_config.max_time)) if train_config.max_time else ''} | "
                elif train_config.max_steps:
                    experiment_print = f"Step: {step:>7d}{('/' + to_readable_format(train_config.max_steps)) if train_config.max_steps else ''} | "
                elif train_config.max_loss:
                    experiment_print = f"Loss: {loss:>7s}{('/' + to_readable_format(train_config.max_loss)) if train_config.max_loss else ''} | "
                else:
                    experiment_print = ""
                logger.info(
                    f"[rank {pgm.process_group_manager.global_rank}] "
                    f"Step: {step:<5d} | "
                    f"Loss: {loss:6.4f} | "
                    f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                    f"Tok/s: {to_readable_format(tokens_per_second):>7s} | "
                    f"Tok/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                    f"{experiment_print}"
                    f"MFU: {mfu:5.2f}% | "
                    f"Memory usage: {torch.cuda.memory_reserved() / 1e9:6.2f}GB | "
                    f"T: {goodput_log['throughput']:6.4f} | "
                    f"SE: {goodput_log['statistical_efficiency']:.4e} | "
                    f"G: {goodput_log['goodput']:6.4f} | "
                    f"Avg T: {goodput_log['goodput']:6.4f} | "
                    f"Avg SE: {goodput_log['goodput']:6.4f} | "
                    f"Avg G: {goodput_log['goodput']:6.4f} | "
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
                        "statistical_efficiency": goodput_log["statistical_efficiency"],
                        "avg_throughput": goodput_log["avg_throughput"],
                        "avg_statistical_efficiency": goodput_log["avg_statistical_efficiency"],
                        "avg_goodput": goodput_log["avg_goodput"],
                    }
                    results = wandb_log
                    wandb.log(wandb_log)

            if step % train_config.save_frequency == 0:
                checkpoint_manager.save_checkpoint(model, optimizer, step, trained_tokens, train_config.checkpoint_path+f"/{step}")
            
            if train_config.max_steps is not None and step >= train_config.max_steps:
                break

            if train_config.max_tokens is not None and trained_tokens >= train_config.max_tokens:
                break

            if train_config.max_time is not None and end_time - start_time >= train_config.max_time:
                break

            if train_config.max_loss is not None and loss <= train_config.max_loss:
                break

            end_time = time.time()

    if len(results) != 0:
        print(results)
        print(len(df))
        results['batch_size'] = train_config.per_device_train_batch_size
        results['max_seq_length'] = train_config.max_seq_length
        df.loc[len(df)] = results
        df.to_csv(csv_path, index=False)

    if global_rank == 0 and train_config.use_wandb:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == '__main__':
    main(**parse_argv())