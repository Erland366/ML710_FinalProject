"""
This docs explain clearly of what is each parameter looks like for FSDP2

This is important to determine what ZeRO stage is used for the model.
"""
import torch
from torch import nn

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch import distributed as dist

class FSDP():
    def __init__(self, model: nn.Module, zero_stage: int):
        self.model = model
        self.zero_stage = zero_stage
        assert self.zero_stage in [2, 3], "Only ZeRO-2 and ZeRO-3 are supported"

        self._apply_fsdp()

    def _apply_fsdp(self):
        mp_policy = MixedPrecisionPolicy()
        for layer_id, transformer_block in enumerate(self.model.model.layers):
            should_reshard = (
                (self.zero_stage == 3) and layer_id < len(self.model.model.layers) - 1
            )

            fully_shard(
                transformer_block,
                # TODO: Correctly do MP
                mp_policy=mp_policy,
                reshard_after_forward=should_reshard,
            )

        fully_shard(
            self.model,
            mp_policy=mp_policy,
            reshard_after_forward=(self.zero_stage == 3)
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class DataParallelNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.require_backward_grad_sync = True
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def synchronize_gradients(self):
        dp_world_size = pgm.process_group_manager.dp_world_size
        if dp_world_size <= 1:
            return

        dp_group = pgm.process_group_manager.dp_group

        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
                param.grad.data.div_(dp_world_size)