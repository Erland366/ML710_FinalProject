import torch
import torch.nn as nn
import torch.nn.functional as F

import picotron.process_group_manager as pgm
from picotron.pipeline_parallel.pp_communications import pipeline_communicate, bidirectional_pipeline_communicate
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

class PipelineParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = LlamaModelPipelineParallel(model, config)
        self.lm_head = model.lm_head if pgm.process_group_manager.pp_is_last_stage else nn.Identity()

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        position_ids= None,
        past_key_values= None,
        inputs_embeds= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
        cache_position= None,
        **flash_attn_kwargs,
    ):
        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )
        return self.lm_head(x.last_hidden_state)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        """
        Backward pass for this pipeline stage.
        Computes gradients for assigned layers using received gradient from next stage.
        """
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        # torch.autograd.backward will automatically accumulates gradients in the leaves (cf: https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

class LlamaModelPipelineParallel(LlamaModel):
    """
    Implements pipeline parallelism by distributing model layers across multiple GPUs.
    Each GPU processes a subset of the model's layers in a pipeline fashion.
    """
    def __init__(self, model, config):
        super().__init__(config)
        # Determine which layers should be assigned to this GPU
        self.layer_distribution = self.distribute_layers(config.num_hidden_layers)
        # Only first stage has embedding layer, others use Identity
        self.embed_tokens = model.model.embed_tokens if pgm.process_group_manager.pp_is_first_stage else nn.Identity() #Changed to embed_tokens
        # Assign relevant decoder layers to this GPU
        self.layers = nn.ModuleDict({str(i): model.model.layers[i] for i in self.layer_distribution}) #Changed to Layers
        # Only last stage has normalization and projection layers
        self.norm = model.model.norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize or reset all model parameters for this pipeline stage."""
        
        if pgm.process_group_manager.pp_is_first_stage:
            self.embed_tokens.reset_parameters()

        for layer in self.layers.values(): 
            layer.input_layernorm.reset_parameters()
            layer.self_attn.reset_parameters() #changed to layer.self_attn
            layer.post_attention_layernorm.reset_parameters()
            layer.mlp.reset_parameters()

        if pgm.process_group_manager.pp_is_last_stage:
            self.norm.reset_parameters()
            self.lm_head.reset_parameters()

    def distribute_layers(self, num_layers):
        """
        Distribute model layers across GPUs as evenly as possible.
        Returns the layer indices that should be processed by this GPU.
        """
        # Calculate layers per GPU, handling uneven distribution
        layers_per_gpu = [num_layers // pgm.process_group_manager.pp_world_size + (1 if i < num_layers % pgm.process_group_manager.pp_world_size else 0) for i in range(pgm.process_group_manager.pp_world_size)]
        # Calculate starting layer for this GPU
        start_layer = sum(layers_per_gpu[:pgm.process_group_manager.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[pgm.process_group_manager.pp_rank]))

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        position_ids= None,
        past_key_values= None,
        inputs_embeds= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
        cache_position= None,
        **flash_attn_kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx in sorted([int(i) for i in self.layers.keys()]):
            decoder_layer = self.layers[str(layer_idx)]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    """
    Implements All-Forward-All-Backward (AFAB) pipeline parallel training.
    First performs all forward passes, then all backward passes sequentially.
    
    Args:
        model: The pipeline parallel model
        data_loader: Iterator providing training batches
        tensor_shapes: Expected shapes of tensors for communication
        device: Device to run computations on
        dtype: Data type for tensors
    """
    logging_loss: torch.float32 = 0.0
    # Store tensors to recreate computation graph during backward pass
    input_tensors, output_tensors = [], []
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1

    for _ in range(data_loader.grad_acc_steps): # All forward passes
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), inputs_embeds=batch["hidden_states"])
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps

        # Save tensors to reconstruct computation graph during backward pass
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    for ith_microbatch in range(data_loader.grad_acc_steps): # All backward passes
        if requires_grad_sync:
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            model.require_backward_grad_sync = is_last_iteration
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        # Retrieve saved tensors in FIFO order to match forward pass sequence
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss

def train_step_pipeline_1f1b(model, data_loader, tensor_shapes, device, dtype):    
    """
    Implements 1F1B (one-forward-one-backward) pipeline parallel training.
    Interleaves forward and backward passes to improve GPU utilization.
    
    Pipeline stages:
    1. Warmup phase: Forward passes to fill pipeline
    2. Steady state: Alternating forward and backward passes
    3. Cooldown phase: Remaining backward passes
    
    Args:
        model: The pipeline parallel model
        data_loader: Iterator providing training batches
        tensor_shapes: Expected shapes of tensors for communication
        device: Device to run computations on
        dtype: Data type for tensors
    """
    # Calculate number of warmup microbatches needed
    num_warmup_microbatches = min(pgm.process_group_manager.pp_world_size - pgm.process_group_manager.pp_rank - 1, data_loader.grad_acc_steps)
    num_microbatches_remaining = data_loader.grad_acc_steps - num_warmup_microbatches
    logging_loss, input_tensors, output_tensors  = 0.0, [], []
    requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
    
    def _forward_step(input_tensor):
        """Helper function to perform a single forward step in the pipeline."""
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids=batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), inputs_embeds=batch["hidden_states"])
        
        # calculate loss on the last stage
        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1, 2), batch["target_ids"].to(device), reduction='mean')
            nonlocal logging_loss
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps
        return output_tensor

    # Warmup Phase: Fill the pipeline with forward passes
    for _ in range(num_warmup_microbatches):
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        output_tensor = _forward_step(input_tensor)
        pipeline_communicate(operation='send_forward', tensor=output_tensor, device=device, dtype=dtype)
        # Store tensors for later backward passes during cooldown phase
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        #TODO: we should call deallocate_output_tensor as in Megatron-LM
        # During pipeline parallel training, we need to save output tensors for the backward pass.
        # However, between producing an output tensor and using it for backprop, the tensor's data
        # sits idle in memory while only its grad_fn is needed for the computational graph.
        # deallocate_output_tensor replaces the tensor's data with a minimal scalar tensor 
        # (cf https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py#L115),
        # dramatically reducing memory usage while preserving the ability to do backprop later.

    # Steady State Phase: Alternate between forward and backward passes
    if num_microbatches_remaining > 0:
        input_tensor = pipeline_communicate(operation='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
    
    #NOTE: Explanation as to how to make DP and PP work together: https://github.com/huggingface/picotron/pull/5#issue-2629838274
    if requires_grad_sync:
        model.require_backward_grad_sync = False

    for ith_microbatch in range(num_microbatches_remaining):  # 1F1B steady state
        is_last_iteration = (ith_microbatch == num_microbatches_remaining - 1)
        output_tensor = _forward_step(input_tensor)
        output_tensor_grad = bidirectional_pipeline_communicate(operation='send_fwd_recv_bwd', send_tensor=output_tensor, recv_shapes=tensor_shapes, device=device, dtype=dtype)
        # Store current tensors for next backward pass
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        # Retrieve oldest tensors for current backward pass (FIFO order)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        
        # Trigger gradient sync on the last microbatch but only when last rank (the one that has num_warmup_microbatches = 0) has finished computing its backward pass.
        if num_warmup_microbatches == 0 and is_last_iteration:
            model.require_backward_grad_sync = True

        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        
        if is_last_iteration:
            input_tensor = None
            pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)
        else:
            input_tensor = bidirectional_pipeline_communicate(operation='send_bwd_recv_fwd', send_tensor=input_tensor_grad, recv_shapes=tensor_shapes, device=device, dtype=dtype)

    # Cooldown Phase: Complete remaining backward passes
    for ith_warmup_microbatches in range(num_warmup_microbatches):
        if requires_grad_sync:
            is_last_iteration = (ith_warmup_microbatches == num_warmup_microbatches - 1)
            model.require_backward_grad_sync = (ith_warmup_microbatches == num_warmup_microbatches - 1)
        # Process remaining stored tensors from warmup phase in FIFO order
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        output_tensor_grad = pipeline_communicate(operation='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communicate(operation='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss