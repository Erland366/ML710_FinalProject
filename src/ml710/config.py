from typing import Literal
from pydantic_config import BaseConfig

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
    max_tokens: int = 100_000
    num_samples: int | None = None

    # How to define this?
    total_train_steps: int = 200

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
    dp_engine: Literal["ddp", "naive", "bucket"] = "bucket"

    # PP settings
    pp_engine: Literal["afab", "1f1b"] = "afab"
