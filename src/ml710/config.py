from pydantic_config import BaseConfig

class ModelConfig(BaseConfig):
    name: str

class TrainConfig(BaseConfig):
    lr: float = 0.01
    seed: int = 3407

    # The one that's affecting the SE etc
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int

    num_samples: int | None = None

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