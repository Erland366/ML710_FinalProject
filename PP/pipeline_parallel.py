import torch
import torch.nn as nn
import torch.distributed as dist
from picotron import PipelineParallel
from picotron.pipeline_parallel import train_step_pipeline_1f1b as schedule_1f1b
from picotron.pipeline_parallel import train_step_pipeline_afab as schedule_afab


class PipelineParallelModel():
    def __init__(self, model: nn.Module, num_stages: int, schedule_type="1f1b"):
        """
        Implements Pipeline Parallelism using Picotron.
        - Splits the model into `num_stages` partitions.
        - Uses either 1F1B or AFAB scheduling.
        """

        self.model = model
        self.num_stages = num_stages
        self.schedule_type = schedule_type
        assert self.num_stages > 1, "Pipeline Parallelism requires at least 2 stages."

        self._apply_pipeline_parallelism()

    def _apply_pipeline_parallelism(self):
        """Splits the model into pipeline stages and applies the selected schedule."""
        world_size = dist.get_world_size()
        assert self.num_stages <= world_size, "Number of pipeline stages exceeds available GPUs!"

        #Scheduling strategy
        if self.schedule_type == "1f1b":
            schedule = schedule_1f1b
        elif self.schedule_type == "afab":
            schedule = schedule_afab
        else:
            raise ValueError("Invalid schedule type. Choose '1f1b' or 'afab'.")

        #Partition model layers across GPUs
        self.model = PipelineParallel(self.model, self.num_stages, schedule=schedule)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
