import os

from typing import Literal
from pydantic_config import BaseConfig
from pydantic import model_validator

class ModelConfig(BaseConfig):
    name: str

class TrainConfig(BaseConfig):
    lr: float = 0.01
    seed: int = 3407
    use_fused_adam: bool = True

    # The one that's affecting the SE etc
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    max_tokens: int | None = None
    max_time: int | None = None
    max_loss: float | None = None

    num_samples: int | None = None

    # How to define this?
    max_steps: int | None = None

    # logging
    use_wandb: bool = True
    run_name: str = ""
    pretrain: bool = False

    # Checkpoint
    checkpoint_path: str = "checkpoints"
    save_frequency: int = 1000
    load_path: str | None = None

    # compile
    use_compile: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"

    # profiling
    run_profile: bool = False
    wait_steps: int | None = None
    warmup_steps: int | None = None
    active_steps: int | None = None
    num_cycles: int | None = None
    profile_path: str = "profile"
    profile_memory: bool = True
    with_stack: bool = True
    record_shapes: bool = True
    with_flops: bool = True

    # Results file name 
    results_file: str | None = None

    @model_validator(mode="after")
    def validate_profile(self):
        if self.run_profile:
            if os.environ["RANK"] == "0":
                print("Profiling is enabled, enable max_steps = 10")
                self.max_steps = 10
                self.max_tokens = None
                self.max_time = None
        return self

class DataConfig(BaseConfig):
    path: str
    num_workers: int = 4
    num_proc: int = 1
    subset_name: str | None = None
    split: str = "train"

class ParallelConfig(BaseConfig):
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    dp_engine: Literal["ddp", "naive", "bucket", "fsdp", "wait_free"] = "bucket"
    zero_stage: int = 3

    # PP settings
    pp_engine: Literal["afab", "1f1b"] = "afab"
    tp_engine: Literal["sync", "async"] = "sync"
