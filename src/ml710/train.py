from ml710.config import TrainConfig, DataConfig
from pydantic_config import parse_argv
from pydantic import validate_call

@validate_call
def main(train_config: TrainConfig, data_config: DataConfig):
    print(train_config)
    print(data_config)

if __name__ == '__main__':
    main(**parse_argv())