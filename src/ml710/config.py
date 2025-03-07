from pydantic_config import BaseConfig

class TrainConfig(BaseConfig):
    lr: float = 0.01

class DataConfig(BaseConfig):
    path: str

class ParallelConfig(BaseConfig):
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1