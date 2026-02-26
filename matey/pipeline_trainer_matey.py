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
    def __init__(self, original_dataset, hierarchical=None):
        self.original_dataset = original_dataset
        self._length = len(original_dataset)
        # For turbt: filter targets down to coarsest level resolution
        self.hierarchical = hierarchical
        if hierarchical is not None:
            from matey.data_utils.utils import construct_filterkernel, construct_filterkernel2D
            self.filtersize = hierarchical["filtersize"]
            self.nhlevels = hierarchical["nlevels"]
            self.datafilter_kernel = construct_filterkernel(self.filtersize)
            self.datafilter_kernel2D = construct_filterkernel2D(self.filtersize)
        
    def __len__(self):
        return self._length
    
    def _filter_to_coarsest(self, x):
        """Apply low-pass filtering (nhlevels-1) times to match coarsest resolution.
        Input x: [T, C, D, H, W] or [C, D, H, W]"""
        if self.hierarchical is None:
            return x
        squeeze = False
        if x.ndim == 4:
            x = x.unsqueeze(0)  # Add T dim
            squeeze = True
        T, C, D, H, W = x.shape
        for _ in range(self.nhlevels - 1):
            x_flat = x.reshape(T * C, D, H, W)
            if D == 1:
                filtered = F.conv3d(x_flat[:, None, :, :, :], self.datafilter_kernel2D,
                                   stride=(1, self.filtersize, self.filtersize))
            else:
                filtered = F.conv3d(x_flat[:, None, :, :, :], self.datafilter_kernel,
                                   stride=self.filtersize)
            filtered = filtered.squeeze(1)  # Remove the c1=1 dim
            _, D, H, W = filtered.shape
            x = filtered.reshape(T, C, D, H, W)
        if squeeze:
            x = x.squeeze(0)
        return x
    
    def __getitem__(self, idx):
        # Ensure idx is a plain Python int — the underlying dataset uses numpy
        # operations (np.searchsorted, subtraction with offsets) that fail if
        # idx is a numpy array or tensor rather than a scalar int.
        if isinstance(idx, (torch.Tensor,)):
            idx = idx.item()
        elif hasattr(idx, '__index__'):
            idx = idx.__index__()
        else:
            idx = int(idx)
        
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
                self._to_tensor(data.get("cond_input", None)),
                self._to_tensor(data.get("cond_field_labels", None)),
                self._to_tensor(data.get("cond_fields", None)),
                torch.tensor([1], dtype=torch.long)  # is_graph=True
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
                self._to_tensor(data.get("cond_input", None)),
                self._to_tensor(data.get("cond_field_labels", None)),
                self._to_tensor(data.get("cond_fields", None)),
                torch.tensor([0], dtype=torch.long)  # is_graph=False
            )
        
        label_data = tar
        # For turbt hierarchical: filter target spatially to coarsest level
        # to match model output resolution. Do NOT index by field_labels — the
        # target already contains only the relevant physical fields (e.g., 4 for
        # isotropic1024fine), while field_labels are global indices into the
        # n_states embedding space.
        if self.hierarchical is not None and not ("graph" in data):
            with torch.no_grad():
                label_data = self._filter_to_coarsest(tar.float())
                # Take last timestep if time dimension present
                if label_data.ndim == 5:  # [T, C, D, H, W]
                    label_data = label_data[-1]  # [C, D_coarse, H_coarse, W_coarse]
        return (input_tuple, label_data)
    
    @staticmethod
    def _to_tensor(val):
        """Convert a value to a tensor. None becomes a single-element zero sentinel."""
        if val is None:
            return torch.tensor([0], dtype=torch.float32)
        if isinstance(val, torch.Tensor):
            return val
        return torch.as_tensor(val)


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
            
            # is_graph was encoded as a tensor for DeepSpeed compatibility
            if torch.is_tensor(is_graph):
                is_graph = is_graph.flatten()[0].item() > 0
            
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
                x_coarsen = self.debed_tokenizer(x_coarsen)
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
        # space_bag and posbias are ModuleLists indexed by level.
        # We use the last (finest) level.
        layers.append(EncodingPipeStage(base_model.space_bag[-1],
                                       encoder_embed_tokenizer,
                                       base_model.posbias[-1] if base_model.posbias is not None and len(base_model.posbias) > 0 else None,
                                       params))
        
        # Transformer stages
        for block in base_model.blocks:
            layers.append(TransformerPipeStage(block))
        
        # Decoding stage
        layers.append(DecodingPipeStage(decoder_debed_tokenizer, coarse_patch_size))
    
    elif params.model_type == 'turbt':
        # TURBT hierarchical model
        #
        # turbt.forward() is recursive: starts at finest level, filters down to coarsest,
        # then processes from coarse → fine with residual connections.
        # The pipeline must replicate this: filter → per-level (encode+blocks+decode) → upsample
        #
        # Pipeline structure for nlevels=3:
        #   Stage 0: DataFilterStage   - filter full-res data down to coarsest resolution
        #   Stage 1: LevelEncodeStage  - level 0 (coarsest): normalize + space_bag + tokenize + posbias
        #   Stage 2-5: TransformerPipeStage - level 0 transformer blocks
        #   Stage 6: LevelDecodeStage  - level 0: de-tokenize
        #   Stage 7: LevelTransitionStage - upsample level 0 → level 1, encode level 1
        #   Stage 8-11: TransformerPipeStage - level 1 blocks
        #   Stage 12: LevelDecodeStage - level 1
        #   Stage 13: LevelTransitionStage - upsample level 1 → level 2, encode level 2
        #   Stage 14-17: TransformerPipeStage - level 2 blocks
        #   Stage 18: FinalDecodeStage - level 2: decode + denormalize + select output fields
        
        nhlevels = params.hierarchical["nlevels"]
        filtersize = params.hierarchical["filtersize"]
        
        tkhead_name = "default"
        if hasattr(params, 'tokenizer_heads') and len(params.tokenizer_heads) > 0:
            tkhead_name = params.tokenizer_heads[0]['head_name']
        
        if global_rank == 0:
            print(f"TURBT pipeline: nhlevels={nhlevels}, filtersize={filtersize}, head='{tkhead_name}'", flush=True)
        
        # ---- Per-level stage classes for turbt ----
        
        class TurbtDataFilterStage(nn.Module):
            """Filter full-resolution input data down to the coarsest level."""
            def __init__(self, datafilter_kernel, datafilter_kernel2D, filtersize, nhlevels):
                super().__init__()
                # Register kernels as buffers so they move with the module
                self.register_buffer('datafilter_kernel', datafilter_kernel)
                self.register_buffer('datafilter_kernel2D', datafilter_kernel2D)
                self.filtersize = filtersize
                self.nhlevels = nhlevels
            
            def forward(self, inputs):
                if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], tuple):
                    inputs, _ = inputs  # Training mode
                
                inp, field_labels, field_labels_out, bcs, leadtime, cond_input, cond_field_labels, cond_fields, is_graph = inputs
                
                if torch.is_tensor(is_graph):
                    is_graph = is_graph.flatten()[0].item() > 0
                
                device = self.datafilter_kernel.device
                model_dtype = self.datafilter_kernel.dtype
                
                inp = inp.to(device, dtype=model_dtype)
                inp = rearrange(inp, 'b t c d h w -> t b c d h w')
                
                # Apply filtering (nhlevels-1) times to go from finest to coarsest
                x = inp
                for _ in range(self.nhlevels - 1):
                    T, B, C, D, H, W = x.shape
                    x_flat = rearrange(x, 't b c d h w -> (t b c) d h w')
                    if D == 1:
                        filtered = F.conv3d(x_flat[:, None, :, :, :], self.datafilter_kernel2D,
                                           stride=(1, self.filtersize, self.filtersize))
                    else:
                        filtered = F.conv3d(x_flat[:, None, :, :, :], self.datafilter_kernel,
                                           stride=self.filtersize)
                    x = rearrange(filtered, '(t b c) c1 d h w -> t b (c c1) d h w', t=T, b=B, c=C)
                
                # Pack metadata as int views for pipeline transport
                bcs_int = bcs.to(device, dtype=model_dtype).view(torch.int32)
                leadtime_int = leadtime.to(device, dtype=model_dtype).view(torch.int32)
                
                # x is now at coarsest resolution, shape [T, B, C, D_coarse, H_coarse, W_coarse]
                # Flatten to [B, ...] for pipeline: store as contiguous tensor
                T, B, C, D, H, W = x.shape
                shape_info = torch.tensor([T, B, C, D, H, W], dtype=torch.long, device=device)
                x_flat = x.contiguous().view(T * B * C * D * H * W)
                
                return (x_flat, shape_info, field_labels.to(device), field_labels_out.to(device),
                        bcs_int, leadtime_int,
                        torch.tensor([is_graph], dtype=torch.bool, device=device))
        
        class TurbtLevelEncodeStage(nn.Module):
            """Normalize, space_bag embed, tokenize, add posbias for one level."""
            def __init__(self, space_bag, embed_tokenizer, posbias, level_idx):
                super().__init__()
                self.space_bag = space_bag
                self.embed_tokenizer = embed_tokenizer
                self.posbias = posbias
                self.level_idx = level_idx
            
            def forward(self, inputs):
                (x_flat, shape_info, field_labels, field_labels_out,
                 bcs_int, leadtime_int, is_graph_t) = inputs
                
                device = next(self.space_bag.parameters()).device
                model_dtype = next(self.space_bag.parameters()).dtype
                
                T, B, C, D, H, W = [s.item() for s in shape_info]
                x = x_flat.to(device, dtype=model_dtype).view(T, B, C, D, H, W)
                
                # Normalize
                x, data_mean, data_std = normalize_spatiotemporal_persample(x)
                
                # Space embedding
                x_pre = rearrange(x, 't b c d h w -> t b d h w c')
                x_pre = self.space_bag(x_pre, field_labels)
                x_pre = rearrange(x_pre, 't b d h w c_emb -> t b c_emb d h w')
                
                # Tokenization
                x_padded_t = rearrange(x_pre, 't b c d h w -> (t b) c d h w')
                x_tok = self.embed_tokenizer(x_padded_t)
                x_tok = rearrange(x_tok, '(t b) c d h w -> t b c d h w', t=T)
                x_tok = rearrange(x_tok, 't b c d h w -> t b c (d h w)')
                
                # Positional encoding
                from functools import reduce
                from operator import mul
                space_dims = [D, H, W]
                ps = self.embed_tokenizer.patch_size
                ntokendim = [dim // p for dim, p in zip(space_dims, ps)]
                delta = [1.0 / dim * p for dim, p in zip(space_dims, ps)]
                
                t_pos_area = torch.zeros(B, T, ntokendim[0], ntokendim[1], ntokendim[2],
                                         2 + len(space_dims), device=device, dtype=model_dtype)
                t_pos_area[..., 0] = repeat(torch.arange(T, device=device),
                                            "t -> b t d h w", b=B, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
                expand_patterns = {0: "d -> b t d h w", 1: "h -> b t d h w", 2: "w -> b t d h w"}
                for i in range(len(ntokendim)):
                    pos = torch.arange(delta[i] * 0.5, 1.0, delta[i], device=device, dtype=model_dtype)
                    t_pos_area[..., i + 1] = repeat(pos, expand_patterns[i],
                                                     b=B, t=T, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
                t_pos_area[..., -1] = reduce(mul, delta)
                
                tposarea_padding = rearrange(t_pos_area, 'b t d h w c -> b t (d h w) c')
                x_seq = rearrange(x_tok, 't b c ntoken_tot -> b c (t ntoken_tot)')
                
                if self.posbias is not None:
                    posbias_val = self.posbias(tposarea_padding, mask_padding=None,
                                               use_zpos=True if D > 1 else False)
                    posbias_val = rearrange(posbias_val, 'b t L c -> b c (t L)')
                    x_seq = x_seq + posbias_val
                
                # Pack data_mean/data_std as int views
                data_mean_int = data_mean.view(torch.int32)
                data_std_int = data_std.view(torch.int32)
                
                return (x_seq, shape_info, field_labels, field_labels_out,
                        bcs_int, leadtime_int, data_mean_int, data_std_int, is_graph_t)
        
        class TurbtLevelDecodeStage(nn.Module):
            """De-tokenize output of transformer blocks for one level.
            Returns spatial output tensor directly (last pipeline stage)."""
            def __init__(self, debed_tokenizer, patch_size, level_idx, nhlevels, state_labels_select=False):
                super().__init__()
                self.debed_tokenizer = debed_tokenizer
                self.patch_size = patch_size
                self.level_idx = level_idx
                self.nhlevels = nhlevels
                self.state_labels_select = state_labels_select  # True only for final level
            
            def forward(self, inputs):
                (x_seq, shape_info, field_labels, field_labels_out,
                 bcs_int, leadtime_int, data_mean_int, data_std_int, is_graph_t) = inputs
                
                T, B, C, D, H, W = [s.item() for s in shape_info]
                
                # Reshape from sequence back to spatial
                x = rearrange(x_seq, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)
                ntokendim = [D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]]
                
                x = rearrange(x, 't b c (d h w) -> t b c d h w',
                             d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
                x = rearrange(x, 't b c d h w -> (t b) c d h w')
                x = self.debed_tokenizer(x)
                x = rearrange(x, '(t b) c d h w -> t b c d h w', t=T)
                
                # x now has n_states_out channels (e.g. 33)
                # Select output fields FIRST (33 -> 4), then denormalize
                # This matches turbt.forward line 341:
                #   x_correct[:,state_labels[0],...] * data_std[-1] + data_mean[-1]
                x_out = x[-1]  # Last timestep: [B, n_states_out, D, H, W]
                
                if self.state_labels_select:
                    x_out = x_out[:, field_labels[0], ...]  # [B, C_sel, D, H, W]
                
                # Now denormalize — data_mean/data_std have C_sel channels
                data_mean = data_mean_int.view(torch.float32).to(x_out.dtype)
                data_std = data_std_int.view(torch.float32).to(x_out.dtype)
                # data_mean shape is [1, B, C, 1, 1, 1] from normalize, squeeze T dim
                data_mean = data_mean.squeeze(0)  # [B, C, 1, 1, 1]
                data_std = data_std.squeeze(0)
                x_out = x_out * data_std + data_mean
                
                return x_out
        
        # ---- Build pipeline for coarsest level only ----
        # For turbt, pipeline parallelism is applied to the coarsest level (level 0)
        # which has the smallest data and most of the compute in the transformer blocks.
        # This is the pragmatic approach since the full multi-level turbt forward is
        # recursive with inter-level residual connections that don't map to a linear pipeline.
        
        imod = 0  # coarsest level
        
        try:
            head_dict = base_model.tokenizer_ensemble_heads[imod][tkhead_name]
            encoder_embed_tokenizer = head_dict["embed"][-1]
            decoder_debed_tokenizer = head_dict["debed"][-1]
            coarse_patch_size = encoder_embed_tokenizer.patch_size
            
            if global_rank == 0:
                print(f"Using coarsest level {imod} for pipeline", flush=True)
                print(f"Patch size: {coarse_patch_size}", flush=True)
                print(f"Transformer blocks at level 0: {len(base_model.module_blocks['0'])}", flush=True)
        except Exception as e:
            if global_rank == 0:
                print(f"Error accessing level {imod} tokenizer: {e}", flush=True)
            raise
        
        # Stage 1: Filter data down to coarsest resolution
        layers.append(TurbtDataFilterStage(
            base_model.datafilter_kernel,
            base_model.datafilter_kernel2D,
            filtersize, nhlevels
        ))
        
        # Stage 2: Encode at coarsest level
        layers.append(TurbtLevelEncodeStage(
            base_model.space_bag[imod],
            encoder_embed_tokenizer,
            base_model.posbias[imod] if base_model.posbias is not None and len(base_model.posbias) > 0 else None,
            imod
        ))
        
        # Stages 3+: Transformer blocks at coarsest level
        level_blocks = base_model.module_blocks[str(imod)]
        if global_rank == 0:
            print(f"  Level {imod}: {len(level_blocks)} transformer blocks", flush=True)
        
        # Wrap transformer blocks — the turbt encode stage outputs a different tuple
        class TurbtTransformerPipeStage(nn.Module):
            def __init__(self, transformer_block):
                super().__init__()
                self.block = transformer_block
            
            def forward(self, inputs):
                (x_seq, shape_info, field_labels, field_labels_out,
                 bcs_int, leadtime_int, data_mean_int, data_std_int, is_graph_t) = inputs
                
                bcs = bcs_int.view(torch.float32).to(x_seq.dtype)
                leadtime = leadtime_int.view(torch.float32).to(x_seq.dtype)
                
                x_seq = self.block(x_seq, bcs=bcs, leadtime=leadtime, mask_padding=None)
                
                return (x_seq, shape_info, field_labels, field_labels_out,
                        bcs_int, leadtime_int, data_mean_int, data_std_int, is_graph_t)
        
        for block in level_blocks:
            layers.append(TurbtTransformerPipeStage(block))
        
        # Final stage: Decode at coarsest level
        layers.append(TurbtLevelDecodeStage(
            decoder_debed_tokenizer, coarse_patch_size, imod, nhlevels,
            state_labels_select=True  # Select output fields at final decode
        ))
        
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
    wrapped_valid_dataset = MATEYPipelineDatasetWrapper(
        valid_dataset,
        hierarchical=getattr(params, 'hierarchical', None) if params.model_type == 'turbt' else None
    )
    
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
    # Get data loaders FIRST — need dataset to auto-correct n_states before model build
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
    
    # Auto-correct n_states if too small for the dataset labels
    # (matches train.py lines 83-87)
    labels_total = [train_dataset.subset_dict[dset] for dset in train_dataset.subset_dict]
    labels_total = [item for sublist in labels_total for item in sublist]
    if params.n_states < max(labels_total) + 1:
        if global_rank == 0:
            print(f"Warning, reserved n_states {params.n_states} is too small for datasets, "
                  f"set it to {max(labels_total)+1} instead")
        params.n_states = max(labels_total) + 1
    
    # Build base model (now with corrected n_states)
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
    # DeepSpeed's PipelineModule constructor calls dist.get_rank() through DS's
    # own comm backend, which must be initialized before PipelineModule is created.
    # deepspeed.init_distributed() (called in basic_usage.py) sets up torch.distributed
    # but not necessarily DeepSpeed's internal comm backend (cdb). We ensure it here.
    if not deepspeed.comm.is_initialized():
        deepspeed.comm.init_distributed(dist_backend='nccl')
    
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
    # The sampler from get_data_loader is a MultisetBatchSampler (a batch sampler
    # that yields lists of indices). It must be passed as batch_sampler, not sampler.
    # batch_size and drop_last are controlled by the batch_sampler itself.
    pipeline_dataset = MATEYPipelineDatasetWrapper(
        train_dataset,
        hierarchical=getattr(params, 'hierarchical', None) if params.model_type == 'turbt' else None
    )
    
    def pipeline_collate(batch):
        """Collate list of (input_tuple, target) into batched (input_tuple, target).
        All elements are tensors (None/bool converted in wrapper).
        """
        input_tuples, targets = zip(*batch)
        
        # Stack targets
        batched_targets = torch.stack(targets, dim=0)
        
        # Stack each element of input_tuple across the batch
        batched_inputs = []
        for i in range(len(input_tuples[0])):
            elems = [t[i] for t in input_tuples]
            batched_inputs.append(torch.stack(elems, dim=0))
        
        return (tuple(batched_inputs), batched_targets)
    
    pipeline_dataloader = DataLoader(
        pipeline_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=pipeline_collate
    )
    
    if global_rank == 0:
        print("Starting MATEY DeepSpeed Pipeline Training Loop...")
    
    train_iter = iter(RepeatingLoader(pipeline_dataloader))
    steps_per_epoch = len(pipeline_dataloader)
    total_steps = params.max_epochs * steps_per_epoch
    
    if global_rank == 0:
        print(f"Total training steps: {total_steps} ({params.max_epochs} epochs)")
    
    # Training tracking
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
            
            # Only the last pipeline stage computes the loss.
            # Log from last-stage ranks only — no cross-stage collectives needed.
            if engine.is_last_stage():
                epoch_duration = time.time() - epoch_start_time
                current_loss = last_loss_tensor.item()
                
                print(f"[Stage {engine.stage_id}] Epoch {current_epoch}/{params.max_epochs} "
                      f"(took {epoch_duration:.2f}s) | Train Loss: {current_loss:.6f}")
                
                # Save checkpoint (DeepSpeed handles multi-stage saving)
                if params.save_checkpoint and global_rank == 0:
                    checkpoint_path = os.path.join(params.experiment_dir,
                                                   f'training_checkpoints/ckpt_epoch{current_epoch}')
                    engine.save_checkpoint(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
            
            epoch_start_time = time.time()
            last_loss_tensor.zero_()
    
    # Final summary
    total_duration = time.time() - total_start_time
    max_allocated_gb = torch.cuda.max_memory_allocated(engine.device) / 1024**3
    
    if engine.is_last_stage() and engine.mpu.get_data_parallel_rank() == 0:
        print(f"\n--- Finished training for {total_steps} steps (Total time: {total_duration:.2f}s) ---")
        print(f"Peak GPU Memory Allocated: {max_allocated_gb:.3f} GB")
        print("\nMATEY Pipeline training completed successfully!")
