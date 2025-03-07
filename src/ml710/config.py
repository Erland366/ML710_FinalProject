from pydantic_config import BaseConfig

class TrainConfig(BaseConfig):
    lr: float = 0.01

class DataConfig(BaseConfig):
    path: str