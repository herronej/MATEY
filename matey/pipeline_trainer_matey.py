# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange, repeat
import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
import time
import copy

from .models.vit import build_vit
from .models.avit import build_avit
from .models.svit import build_svit
from .models.turbt import build_turbt
from .data_utils.datasets import get_data_loader
from .data_utils.shared_utils import normalize_spatiotemporal_persample
from .utils.forward_options import ForwardOptionsBase
from .utils.training_utils import compute_loss_and_logs, update_loss_logs_inplace_eval
from .utils.distributed_utils import determine_turt_levels

class MATEYPipelineDatasetWrapper(Dataset):
    """
    Wrapper for MATEY datasets to format data for DeepSpeed PipelineModule.
    Handles both grid-based and graph-based data.
    """
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self._length = len(original_dataset)
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        data = self.original_dataset[idx]
        
        # Handle graph data
        if "graph" in data:
            graphdata = data["graph"]
            tar = graphdata.y
            leadtime = graphdata.leadtime
            
            input_tuple = (
                graphdata,  # graph structure
                data["field_labels"],
                data["field_labels_out"],
                data["bcs"],
                leadtime,
                data.get("cond_input", None),
                data.get("cond_field_labels", None),
                data.get("cond_fields", None),
                True  # is_graph flag
            )
        else:
            # Grid-based data
            inp = data["input"]
            if not isinstance(inp, torch.Tensor):
                inp = torch.from_numpy(inp).float()
            
            tar = data["label"]
            if not isinstance(tar, torch.Tensor):
                tar = torch.from_numpy(tar).float()
            
            field_labels = data["field_labels"]
            if not isinstance(field_labels, torch.Tensor):
                field_labels = torch.tensor(field_labels, dtype=torch.long)
            
            bcs = data["bcs"]
            if not isinstance(bcs, torch.Tensor):
                bcs = torch.tensor(bcs, dtype=torch.float32)
            
            leadtime = data["leadtime"]
            if leadtime is not None and not isinstance(leadtime, torch.Tensor):
                leadtime = torch.tensor([leadtime] if isinstance(leadtime, (int, float)) else leadtime, dtype=torch.float32)
            elif leadtime is None:
                leadtime = torch.tensor([1.0], dtype=torch.float32)
            
            input_tuple = (
                inp,
                field_labels,
                field_labels,  # field_labels_out (same as input for grid)
                bcs,
                leadtime,
                data.get("cond_input", None),
                data.get("cond_field_labels", None),
                data.get("cond_fields", None),
                False  # is_graph flag
            )
        
        label_data = tar
        return (input_tuple, label_data)


def create_matey_pipeline(base_model, params, global_rank=0):
    """
    Creates pipeline stages for MATEY models using tuple + bit-level view method.
    Supports: ViT variants (vit_all2all, avit, svit) and TURBT (hierarchical turbulence model)
    """
    layers = []
    
    class EncodingPipeStage(nn.Module):
        def __init__(self, space_bag, embed_tokenizer, posbias, params):
            super().__init__()
            self.space_bag = space_bag
            self.embed_tokenizer = embed_tokenizer
            self.posbias = posbias
            self.params = params
        
        def forward(self, inputs):
            # Unpack inputs
            if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], tuple):
                inputs, _ = inputs  # Training mode
            
            inp, field_labels, field_labels_out, bcs, leadtime, cond_input, cond_field_labels, cond_fields, is_graph = inputs
            
            device = next(self.space_bag.parameters()).device
            model_dtype = next(self.space_bag.parameters()).dtype
            
            # Move to device and cast dtype
            if not is_graph:
                inp = inp.to(device, dtype=model_dtype)
                inp = rearrange(inp, 'b t c d h w -> t b c d h w')
                T, B, C, D, H, W = inp.shape
                
                # Normalize
                x, data_mean, data_std = normalize_spatiotemporal_persample(inp)
                
                # Space embedding
                x_pre = rearrange(x, 't b c d h w -> t b d h w c')
                x_pre = self.space_bag(x_pre, field_labels.to(device))
                x_pre = rearrange(x_pre, 't b d h w c_emb -> t b c_emb d h w')
                
                # Tokenization
                x_padded_t = rearrange(x_pre, 't b c d h w -> (t b) c d h w')
                x_padding = self.embed_tokenizer(x_padded_t)
                x_padding = rearrange(x_padding, '(t b) c d h w -> t b c d h w', t=T)
                x_padding = rearrange(x_padding, 't b c d h w -> t b c (d h w)')
                
                # Positional encoding
                from functools import reduce
                from operator import mul
                space_dims = x.shape[3:]
                ps = self.embed_tokenizer.patch_size
                ntokendim = [dim // p for dim, p in zip(space_dims, ps)]
                delta = [1.0/dim*p for dim, p in zip(space_dims, ps)]
                
                t_pos_area = torch.zeros(B, T, ntokendim[0], ntokendim[1], ntokendim[2], 
                                        2 + len(space_dims), device=device, dtype=model_dtype)
                t_pos_area[..., 0] = repeat(torch.arange(T, device=device), "t -> b t d h w", 
                                           b=B, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
                
                expand_patterns = {0: "d -> b t d h w", 1: "h -> b t d h w", 2: "w -> b t d h w"}
                for i, dim_len in enumerate(ntokendim):
                    pos = torch.arange(delta[i] * 0.5, 1.0, delta[i], device=device, dtype=model_dtype)
                    t_pos_area[..., i + 1] = repeat(pos, expand_patterns[i], 
                                                    b=B, t=T, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
                t_pos_area[..., -1] = reduce(mul, delta)
                
                tposarea_padding = rearrange(t_pos_area, 'b t d h w c-> b t (d h w) c')
                x_padding = rearrange(x_padding, 't b c ntoken_tot -> b c (t ntoken_tot)')
                
                if self.posbias is not None:
                    posbias = self.posbias(tposarea_padding, mask_padding=None, use_zpos=True if D > 1 else False)
                    posbias = rearrange(posbias, 'b t L c -> b c (t L)')
                    x_padding = x_padding + posbias
                
                shape_info = torch.tensor([T, D, H, W], dtype=torch.long, device=device)
                
                # Bit-level view for metadata
                bcs_int_view = bcs.to(device, dtype=model_dtype).view(torch.int32)
                data_mean_int_view = data_mean.view(torch.int32)
                data_std_int_view = data_std.view(torch.int32)
                leadtime_int_view = leadtime.to(device, dtype=model_dtype).view(torch.int32)
                
            else:
                # Graph data path
                x_padding = inp.to(device)  # Graph object
                shape_info = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)  # Dummy for graphs
                bcs_int_view = bcs.to(device, dtype=model_dtype).view(torch.int32)
                # For graphs, no normalization stats
                data_mean_int_view = torch.zeros(1, device=device, dtype=torch.int32)
                data_std_int_view = torch.ones(1, device=device, dtype=torch.int32)
                leadtime_int_view = leadtime.to(device, dtype=model_dtype).view(torch.int32)
            
            return (
                x_padding.contiguous() if not is_graph else x_padding,
                bcs_int_view,
                field_labels.to(device),
                field_labels_out.to(device),
                data_mean_int_view,
                data_std_int_view,
                leadtime_int_view,
                shape_info,
                torch.tensor([is_graph], dtype=torch.bool, device=device)
            )
    
    class TransformerPipeStage(nn.Module):
        def __init__(self, transformer_block):
            super().__init__()
            self.block = transformer_block
        
        def forward(self, inputs):
            features, bcs_int_view, field_labels, field_labels_out, data_mean_int_view, \
                data_std_int_view, leadtime_int_view, shape_info, is_graph_tensor = inputs
            
            bcs = bcs_int_view.view(torch.float32).to(features.dtype)
            leadtime = leadtime_int_view.view(torch.float32).to(features.dtype)
            
            # Process through transformer block
            processed_features = self.block(features, bcs, leadtime=leadtime, mask_padding=None)
            
            return (processed_features, bcs_int_view, field_labels, field_labels_out,
                   data_mean_int_view, data_std_int_view, leadtime_int_view, shape_info, is_graph_tensor)
    
    class DecodingPipeStage(nn.Module):
        def __init__(self, debed_tokenizer, coarse_patch_size):
            super().__init__()
            self.debed_tokenizer = debed_tokenizer
            self.coarse_patch_size = coarse_patch_size
        
        def forward(self, inputs):
            features, bcs_int_view, field_labels, field_labels_out, data_mean_int_view, \
                data_std_int_view, leadtime_int_view, shape_info, is_graph_tensor = inputs
            
            is_graph = is_graph_tensor.item()
            
            if not is_graph:
                # Grid data decoding
                data_mean = data_mean_int_view.view(torch.float32).to(features.dtype)
                data_std = data_std_int_view.view(torch.float32).to(features.dtype)
                
                T, D, H, W = shape_info
                
                x_padding = rearrange(features, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T.item())
                
                from functools import reduce
                from operator import mul
                space_dims = [D.item(), H.item(), W.item()]
                ntokendim = [dim // p for dim, p in zip(space_dims, self.coarse_patch_size)]
                
                x_coarsen = rearrange(x_padding, 't b c (d h w) -> t b c d h w', 
                                     d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
                x_coarsen = rearrange(x_coarsen, 't b c d h w -> (t b) c d h w')
                x_coarsen = self.debed_tokenizer(x_coarsen, field_labels[0])
                x = rearrange(x_coarsen, '(t b) c d h w -> t b c d h w', t=T.item())
                
                # Denormalize
                x = x * data_std + data_mean
                
                return x[-1]  # Return last timestep
            else:
                # Graph data - features are already in correct format
                return features
    
    # ====================================================================
    # Model-specific pipeline construction
    # ====================================================================
    
    if params.model_type in ['vit_all2all', 'avit', 'svit']:
        # Standard ViT-based models
        tkhead_name = "default"
        encoder_embed_tokenizer = base_model.tokenizer_ensemble_heads[-1][tkhead_name]["embed"][-1]
        decoder_debed_tokenizer = base_model.tokenizer_ensemble_heads[-1][tkhead_name]["debed"][-1]
        coarse_patch_size = encoder_embed_tokenizer.patch_size
        
        # Encoding stage
        layers.append(EncodingPipeStage(base_model.space_bag,
                                       encoder_embed_tokenizer,
                                       base_model.posbias,
                                       params))
        
        # Transformer stages
        for block in base_model.blocks:
            layers.append(TransformerPipeStage(block))
        
        # Decoding stage
        layers.append(DecodingPipeStage(decoder_debed_tokenizer, coarse_patch_size))
    
    elif params.model_type == 'turbt':
        # TURBT hierarchical model
        
        # tokenizer_ensemble_heads is a ModuleList, not ModuleDict
        if global_rank == 0:
            print(f"Tokenizer ensemble heads type: {type(base_model.tokenizer_ensemble_heads)}", flush=True)
            print(f"Number of tokenizer heads: {len(base_model.tokenizer_ensemble_heads)}", flush=True)
        
        # Determine which tokenizer head to use
        head_idx = 0  # Default to first head
        if hasattr(params, 'tokenizer_heads') and len(params.tokenizer_heads) > 0:
            tkhead_name = params.tokenizer_heads[0]['head_name']
            if isinstance(tkhead_name, int):
                head_idx = tkhead_name
            elif tkhead_name != 'default':
                try:
                    head_idx = int(tkhead_name)
                except (ValueError, TypeError):
                    if global_rank == 0:
                        print(f"Warning: Could not parse head_name '{tkhead_name}', using index 0", flush=True)
        
        if global_rank == 0:
            print(f"Using tokenizer head index: {head_idx}", flush=True)
        
        # Get the tokenizer head by index
        try:
            tokenizer_head = base_model.tokenizer_ensemble_heads[head_idx]
            
            if global_rank == 0:
                print(f"Tokenizer head type: {type(tokenizer_head)}", flush=True)
            
            # Inspect what's actually in the tokenizer_head
            if isinstance(tokenizer_head, nn.ModuleDict):
                actual_keys = list(tokenizer_head.keys())
                if global_rank == 0:
                    print(f"Tokenizer head keys: {actual_keys}", flush=True)
                
                # Try to find encoder and decoder components with various naming conventions
                encoder_key = None
                decoder_key = None
                
                for key in actual_keys:
                    key_lower = key.lower()
                    if 'embed' in key_lower or 'encode' in key_lower or 'enc' in key_lower:
                        encoder_key = key
                    if 'debed' in key_lower or 'decode' in key_lower or 'dec' in key_lower or 'output' in key_lower:
                        decoder_key = key
                
                if encoder_key is None or decoder_key is None:
                    if global_rank == 0:
                        print(f"Could not find encoder/decoder keys. Available keys: {actual_keys}", flush=True)
                        print("Assuming keys are in order: first for encoder, last for decoder", flush=True)
                    encoder_key = actual_keys[0]
                    decoder_key = actual_keys[-1]
                
                if global_rank == 0:
                    print(f"Using encoder key: '{encoder_key}', decoder key: '{decoder_key}'", flush=True)
                
                embed_tokenizers = tokenizer_head[encoder_key]
                debed_tokenizers = tokenizer_head[decoder_key]
                
            elif hasattr(tokenizer_head, '__dict__'):
                # Try attribute-based access
                if global_rank == 0:
                    print(f"Tokenizer head attributes: {list(vars(tokenizer_head).keys())}", flush=True)
                
                # Try common attribute names
                if hasattr(tokenizer_head, 'embed'):
                    embed_tokenizers = tokenizer_head.embed
                    debed_tokenizers = tokenizer_head.debed
                elif hasattr(tokenizer_head, 'encoder'):
                    embed_tokenizers = tokenizer_head.encoder
                    debed_tokenizers = tokenizer_head.decoder
                else:
                    raise AttributeError(f"Cannot find encoder/decoder in tokenizer_head with attributes: {list(vars(tokenizer_head).keys())}")
            else:
                raise TypeError(f"Unexpected tokenizer_head structure: {type(tokenizer_head)}")
            
            # Get the actual tokenizer modules
            if isinstance(embed_tokenizers, (nn.ModuleList, list)):
                if global_rank == 0:
                    print(f"Number of embed tokenizers: {len(embed_tokenizers)}", flush=True)
                encoder_embed_tokenizer = embed_tokenizers[-1]  # Use last/finest resolution
                decoder_debed_tokenizer = debed_tokenizers[-1]
            else:
                # Single module
                encoder_embed_tokenizer = embed_tokenizers
                decoder_debed_tokenizer = debed_tokenizers
            
            if global_rank == 0:
                print(f"Encoder tokenizer type: {type(encoder_embed_tokenizer)}", flush=True)
                print(f"Decoder tokenizer type: {type(decoder_debed_tokenizer)}", flush=True)
            
            coarse_patch_size = encoder_embed_tokenizer.patch_size
            if global_rank == 0:
                print(f"Patch size: {coarse_patch_size}", flush=True)
            
        except Exception as e:
            if global_rank == 0:
                print(f"Error accessing tokenizer heads: {e}", flush=True)
                import traceback
                traceback.print_exc()
            raise
        
        # Encoding stage
        layers.append(EncodingPipeStage(
            base_model.space_bag,
            encoder_embed_tokenizer,
            base_model.posbias,
            params
        ))
        
        # Transformer blocks
        if global_rank == 0:
            print(f"Processing transformer blocks...", flush=True)
        
        if hasattr(base_model, 'hierarchical_blocks'):
            if global_rank == 0:
                print(f"Using hierarchical_blocks: {type(base_model.hierarchical_blocks)}", flush=True)
            # If model explicitly stores hierarchical blocks
            for level_idx, level_blocks in enumerate(base_model.hierarchical_blocks):
                if isinstance(level_blocks, (list, nn.ModuleList)):
                    if global_rank == 0:
                        print(f"  Level {level_idx}: {len(level_blocks)} blocks", flush=True)
                    for block in level_blocks:
                        layers.append(TransformerPipeStage(block))
                else:
                    # Single block at this level
                    if global_rank == 0:
                        print(f"  Level {level_idx}: single block", flush=True)
                    layers.append(TransformerPipeStage(level_blocks))
        
        elif hasattr(base_model, 'blocks'):
            if global_rank == 0:
                print(f"Using blocks: {type(base_model.blocks)}", flush=True)
            # Fallback: treat as sequential blocks
            if isinstance(base_model.blocks, (list, nn.ModuleList)):
                if global_rank == 0:
                    print(f"  Total blocks: {len(base_model.blocks)}", flush=True)
                for idx, block in enumerate(base_model.blocks):
                    layers.append(TransformerPipeStage(block))
            else:
                raise AttributeError("base_model.blocks is not iterable")
        
        else:
            # Last resort: try to find blocks by inspection
            if global_rank == 0:
                print("Warning: Could not find 'blocks' or 'hierarchical_blocks' attribute", flush=True)
                # Print non-private attributes
                attrs = [attr for attr in dir(base_model) if not attr.startswith('_')]
                print(f"Available attributes: {attrs[:20]}...", flush=True)  # Print first 20 to avoid spam
            raise AttributeError("TURBT model does not have expected 'blocks' or 'hierarchical_blocks' attribute")
        
        if global_rank == 0:
            print(f"Total pipeline layers before decoding: {len(layers)}", flush=True)
        
        # Decoding stage
        layers.append(DecodingPipeStage(decoder_debed_tokenizer, coarse_patch_size))
        
        if global_rank == 0:
            print(f"Total pipeline layers: {len(layers)}", flush=True)
    
    else:
        raise ValueError(f"Unknown model type for pipeline: {params.model_type}")
    
    return layers


def validate_with_pipeline_matey(engine, params, global_rank, world_size, valid_dataset):
    """
    Validation loop for MATEY pipeline.
    """
    engine.eval()
    if global_rank == 0:
        print("\n--- Starting MATEY Validation Phase ---")
    
    data_parallel_size = engine.dp_world_size
    dp_rank = engine.mpu.get_data_parallel_rank()
    
    val_sampler = DistributedSampler(valid_dataset, num_replicas=data_parallel_size, 
                                     rank=dp_rank, shuffle=False)
    wrapped_valid_dataset = MATEYPipelineDatasetWrapper(valid_dataset)
    
    valid_dataloader = DataLoader(
        wrapped_valid_dataset,
        batch_size=params.batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    device = engine.device
    total_nrmse = torch.tensor(0.0, device=device)
    total_l1 = torch.tensor(0.0, device=device)
    total_rmse = torch.tensor(0.0, device=device)
    step_count = 0
    
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            outputs = engine.eval_batch(iter([data]))
            _, targets = data
            targets = targets.to(outputs.device).float()
            outputs = outputs.float()
            
            # Handle graph vs grid data
            if targets.dim() == 2:  # Graph data [nnodes, C]
                residuals = outputs - targets
                nrmse = (residuals.pow(2).mean() / (1e-7 + targets.pow(2).mean())).sqrt()
                l1 = F.l1_loss(outputs, targets)
                rmse = residuals.pow(2).mean().sqrt()
            else:  # Grid data [B, C, D, H, W]
                spatial_dims = tuple(range(outputs.ndim))[2:]
                residuals = outputs - targets
                tar_norm = 1e-7 + targets.pow(2).mean(spatial_dims, keepdim=True)
                nrmse = (residuals.pow(2).mean(spatial_dims, keepdim=True) / tar_norm).sqrt().mean()
                l1 = F.l1_loss(outputs, targets)
                rmse = residuals.pow(2).mean(spatial_dims).sqrt().mean()
            
            total_nrmse += nrmse
            total_l1 += l1
            total_rmse += rmse
            step_count += 1
            
            if i >= 5:  # Early stopping for validation
                break
    
    # Average across steps and DP ranks
    if step_count > 0:
        avg_nrmse = total_nrmse / step_count
        avg_l1 = total_l1 / step_count
        avg_rmse = total_rmse / step_count
    else:
        avg_nrmse = torch.tensor(0.0, device=device)
        avg_l1 = torch.tensor(0.0, device=device)
        avg_rmse = torch.tensor(0.0, device=device)
    
    dist.all_reduce(avg_nrmse, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_l1, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_rmse, op=dist.ReduceOp.SUM)
    
    final_nrmse = avg_nrmse / data_parallel_size
    final_l1 = avg_l1 / data_parallel_size
    final_rmse = avg_rmse / data_parallel_size
    
    engine.train()
    
    if global_rank == 0:
        print(f"--- Validation Complete ---")
    
    return {
        'valid_nrmse': final_nrmse,
        'valid_l1': final_l1,
        'valid_rmse': final_rmse
    }


def train_with_pipeline_matey(params, global_rank, local_rank, world_size):
    """
    Main training function with DeepSpeed pipeline parallelism for MATEY.
    """
    # Build base model
    if params.model_type == 'avit':
        base_model = build_avit(params)
    elif params.model_type == "svit":
        base_model = build_svit(params)
    elif params.model_type == "vit_all2all":
        base_model = build_vit(params)
    elif params.model_type == "turbt":
        base_model = build_turbt(params)
    else:
        raise ValueError(f"Unknown model type: {params.model_type}")
    
    # Get data loaders
    _, train_dataset, sampler = get_data_loader(
        params, params.train_data_paths, 
        dist.is_initialized(), split='train',
        train_offset=params.embedding_offset,
        group_size=1, global_rank=global_rank, num_sp_groups=world_size
    )
    
    _, valid_dataset, _ = get_data_loader(
        params, params.valid_data_paths,
        dist.is_initialized(), split='val',
        group_size=1, global_rank=global_rank, num_sp_groups=world_size
    )
    
    if global_rank == 0:
        print("Creating MATEY pipeline with tuple-based method...")
    
    # Create pipeline layers - NOW PASSING global_rank
    layers = create_matey_pipeline(base_model, params, global_rank)
    
    # Define loss function
    def matey_loss_fn(outputs, targets):
        outputs = outputs.float()
        targets = targets.to(outputs.device)
        
        if targets.dim() == 2:  # Graph data
            residuals = outputs - targets
            tar_norm = 1e-7 + targets.pow(2).mean()
            raw_loss = residuals.pow(2).mean() / tar_norm
        else:  # Grid data
            spatial_dims = tuple(range(outputs.ndim))[2:]
            residuals = outputs - targets
            tar_norm = 1e-7 + targets.pow(2).mean(dim=spatial_dims, keepdim=True)
            raw_loss = (residuals.pow(2).mean(dim=spatial_dims, keepdim=True)) / tar_norm
        
        return raw_loss.mean()
    
    # Create pipeline model
    pipeline_model = PipelineModule(
        layers=layers,
        loss_fn=matey_loss_fn,
        num_stages=params.pipeline_stages,
        partition_method='parameters'
    )
    
    # Calculate batch sizes
    data_parallel_size = world_size // params.pipeline_stages
    global_batch_size = params.batch_size * data_parallel_size
    micro_batch_per_gpu = params.batch_size
    
    # DeepSpeed config
    ds_config = {
        "train_batch_size": global_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_per_gpu,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": params.learning_rate,
                "weight_decay": params.weight_decay if hasattr(params, 'weight_decay') else 0.0,
                "torch_adam": True
            }
        },
        "steps_per_print": 10,
        "pipeline": {
            "pipe_partitioned": True,
            "grad_partitioned": True
        },
        "dataloader_drop_last": True
    }
    
    # Add FP16/BF16 if requested
    if params.enable_amp:
        if torch.cuda.is_bf16_supported():
            ds_config["bf16"] = {"enabled": True}
        else:
            ds_config["fp16"] = {
                "enabled": True,
                "initial_scale_power": 12
            }
    
    # Add ZeRO if requested
    if params.zero_stage > 0:
        ds_config["zero_optimization"] = {
            "stage": params.zero_stage,
            "reduce_bucket_size": 2.5e7
        }
    
    # Initialize DeepSpeed
    engine, _, _, _ = deepspeed.initialize(
        model=pipeline_model,
        config=ds_config,
        training_data=None
    )
    
    # Prepare data
    pipeline_dataset = MATEYPipelineDatasetWrapper(train_dataset)
    pipeline_dataloader = DataLoader(
        pipeline_dataset,
        batch_size=micro_batch_per_gpu,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    if global_rank == 0:
        print("Starting MATEY DeepSpeed Pipeline Training Loop...")
    
    train_iter = iter(RepeatingLoader(pipeline_dataloader))
    steps_per_epoch = len(pipeline_dataloader)
    total_steps = params.max_epochs * steps_per_epoch
    
    if global_rank == 0:
        print(f"Total training steps: {total_steps} ({params.max_epochs} epochs)")
    
    # Training tracking
    if global_rank == 0:
        epoch_losses = []
        epoch_times = []
        validation_nrmse_history = []
    
    last_loss_tensor = torch.tensor(0.0, device=engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    total_start_time = time.time()
    epoch_start_time = time.time()
    
    # Training loop
    for step in range(total_steps):
        loss = engine.train_batch(data_iter=train_iter)
        
        if loss is not None:
            if (engine.global_steps) % ds_config['steps_per_print'] == 0:
                print(f"Step {engine.global_steps}/{total_steps}, Loss: {loss.item():.6f}")
            last_loss_tensor.copy_(loss.detach())
        
        # End of epoch
        if (step + 1) % steps_per_epoch == 0:
            current_epoch = (step + 1) // steps_per_epoch
            
            # Broadcast loss from last pipeline stage
            if params.pipeline_stages > 1:
                dist.broadcast(last_loss_tensor, src=engine.grid.get_last_stage_global_rank())
            
            # Run validation
            valid_logs = validate_with_pipeline_matey(engine, params, global_rank, world_size, valid_dataset)
            
            if global_rank == 0:
                epoch_duration = time.time() - epoch_start_time
                epoch_times.append(epoch_duration)
                current_loss = last_loss_tensor.item()
                epoch_losses.append(current_loss)
                
                if 'valid_nrmse' in valid_logs:
                    validation_nrmse_history.append(valid_logs['valid_nrmse'].item())
                
                print(f"--- Finished Epoch {current_epoch}/{params.max_epochs} (took {epoch_duration:.2f}s) ---")
                if 'valid_nrmse' in valid_logs:
                    print(f"    Train Loss: {current_loss:.6f} | Valid NRMSE: {valid_logs['valid_nrmse'].item():.6f}")
                
                # Save checkpoint
                if params.save_checkpoint:
                    # Save DeepSpeed checkpoint
                    checkpoint_path = os.path.join(params.experiment_dir, 
                                                  f'training_checkpoints/ckpt_epoch{current_epoch}')
                    engine.save_checkpoint(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
            
            epoch_start_time = time.time()
            last_loss_tensor.zero_()
    
    # Final summary
    total_duration = time.time() - total_start_time
    max_allocated_gb = torch.cuda.max_memory_allocated(engine.device) / 1024**3
    mem_tensor = torch.tensor([max_allocated_gb], device=engine.device)
    
    if world_size > 1:
        if global_rank == 0:
            all_mems = [torch.zeros_like(mem_tensor) for _ in range(world_size)]
            dist.gather(mem_tensor, all_mems, dst=0)
        else:
            dist.gather(mem_tensor, [], dst=0)
    else:
        all_mems = [mem_tensor]
    
    if global_rank == 0:
        print(f"\n--- Finished training for {total_steps} steps (Total time: {total_duration:.2f}s) ---")
        print("\n--- Training Summary ---")
        print(f"epoch_losses = {epoch_losses}")
        print(f"validation_nrmse_history = {validation_nrmse_history}")
        print(f"epoch_times = {epoch_times}")
        mem_list = [f"Rank {i}: {mem.item():.3f} GB" for i, mem in enumerate(all_mems)]
        print(f"Peak GPU Memory Allocated per Rank (GB): {', '.join(mem_list)}")
        print("\nMATEY Pipeline training completed successfully!")
