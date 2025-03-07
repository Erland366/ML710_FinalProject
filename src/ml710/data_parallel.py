import torch
from torch import nn

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

class FSDP():
    def __init__(self, model: nn.Module, zero_stage: int):
        self.model = model
        self.zero_stage = zero_stage

        _apply_fsdp()

    def _apply_fsdp(self):
        mp_policy = MixedPrecisionPolicy()
        for layer_id, transformer_block in enumerate(self.model.model.layers):
            should_reshard = (
                (zero_stage == 3) and layer_id < len(self.model.model.layers) - 1
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
            reshard_after_forward=(zero_stage == 3)
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)