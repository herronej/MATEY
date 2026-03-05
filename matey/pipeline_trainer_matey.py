# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
#
# Pipeline trainer for MATEY.
#
# DESIGN RATIONALE
# ================
# The turbt model's forward pass is *recursive* across hierarchical levels
# with cross-level residual connections (filter → recurse → upsample → add).
# It also relies on methods from the base class (get_patchsequence,
# get_spatiotemporalfromsequence, sequence_factor_short/long, ltimeMLP, etc.)
# that are tightly coupled to internal state.
#
# Decomposing this into per-level DeepSpeed PipelineModule stages requires
# faithfully reimplementing ~200 lines of coupled logic and is extremely
# error-prone (the previous attempt was missing ltimeMLP, local_att,
# proper encode/decode paths, etc., causing 5-8x loss regression).
#
# Instead, we use a TWO-STAGE approach:
#   Stage 0: TurbtPreprocessStage  — data reformatting + move to device
#   Stage 1: TurbtModelStage       — calls the ORIGINAL turbt.forward()
#
# This gives us:
#   ✓ Correctness — identical computation to DDP, no reimplementation bugs
#   ✓ Pipeline micro-batch overlap — DeepSpeed still overlaps forward/backward
#     of different micro-batches across the 2 stages
#   ✓ Memory savings — ZeRO-1/2/3 can be combined with the pipeline
#   ✓ Gradient checkpointing — the turbt model already supports this
#
# For DEEPER pipeline parallelism (more stages), the model's transformer
# blocks can be split across stages using DeepSpeed's built-in activation
# checkpoint boundaries, but only after the correctness baseline is
# established.

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange, repeat
import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
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


# ============================================================================
# Dataset wrapper
# ============================================================================

class MATEYPipelineDatasetWrapper(Dataset):
    """
    Wrapper for MATEY datasets to format data for DeepSpeed PipelineModule.
    Targets at FULL resolution — the model runs all hierarchical levels.
    """
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self._length = len(original_dataset)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if isinstance(idx, (torch.Tensor,)):
            idx = idx.item()
        elif hasattr(idx, '__index__'):
            idx = idx.__index__()
        else:
            idx = int(idx)

        data = self.original_dataset[idx]

        if "graph" in data:
            raise NotImplementedError(
                "Graph data not yet supported in pipeline mode. Use DDP.")

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
            leadtime = torch.tensor(
                [leadtime] if isinstance(leadtime, (int, float)) else leadtime,
                dtype=torch.float32)
        elif leadtime is None:
            leadtime = torch.tensor([1.0], dtype=torch.float32)

        input_tuple = (inp, field_labels, bcs, leadtime)

        # Target: last timestep if time dim present
        label_data = tar
        if label_data.ndim == 5:  # [T, C, D, H, W]
            label_data = label_data[-1]  # [C, D, H, W]
        return (input_tuple, label_data)


# ============================================================================
# Pipeline stages
# ============================================================================

class TurbtPreprocessStage(nn.Module):
    """
    Stage 0: Reformat input data and move to device.
    This is a lightweight stage that makes the data ready for the model.
    No trainable parameters — all the parameters live in TurbtModelStage.
    """
    def __init__(self, nhlevels):
        super().__init__()
        self.nhlevels = nhlevels
        # DeepSpeed ZeRO requires at least one trainable parameter per stage,
        # otherwise flatten_dense_tensors gets an empty list and crashes.
        # This tiny parameter has negligible impact on training.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs):
        # Unpack from DeepSpeed: ((input_tuple,), target) in training
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], tuple):
            inputs, _ = inputs

        inp, field_labels, bcs, leadtime = inputs
        device = self.dummy.device
        dtype = torch.float32

        inp = inp.to(device, dtype=dtype)
        # inp shape: [B, T, C, D, H, W]
        inp = rearrange(inp, 'b t c d h w -> t b c d h w')

        T, B, C, D, H, W = inp.shape

        field_labels = field_labels.to(device)
        bcs = bcs.to(device, dtype=dtype)
        leadtime = leadtime.to(device, dtype=dtype)

        # Pack EVERYTHING into a single flat tensor so DeepSpeed tracks only one
        # activation buffer. This avoids "buffer.grad is None" assertions on
        # non-differentiable metadata tensors.
        #
        # CRITICAL: The packed tensor must have FIXED SIZE across all micro-batches.
        # DeepSpeed caches P2P buffer shapes from the first forward pass and reuses
        # them. If the size varies (e.g., different bcs lengths), the P2P recv
        # allocates the wrong buffer → NCCL hang.
        #
        # Layout: [header(4) | shape_info(6) | field_labels(MAX_FL) | bcs(MAX_BCS) | leadtime(MAX_LT) | inp_flat(...)]
        # Fixed metadata region = 4 + 6 + MAX_FL + MAX_BCS + MAX_LT
        MAX_FL = 256    # max field labels (n_states can be up to ~218)
        MAX_BCS = 64    # max bcs elements
        MAX_LT = 16     # max leadtime elements

        shape_f = torch.tensor([T, B, C, D, H, W], dtype=dtype, device=device)

        fl_raw = field_labels[0].to(dtype=dtype)  # 1D labels for first batch elem
        fl_padded = torch.zeros(MAX_FL, dtype=dtype, device=device)
        fl_padded[0] = fl_raw.numel()  # store count in first element
        fl_padded[1:1 + fl_raw.numel()] = fl_raw

        bcs_flat = bcs.view(-1)
        bcs_padded = torch.zeros(MAX_BCS, dtype=dtype, device=device)
        bcs_padded[0] = bcs_flat.numel()
        bcs_padded[1:1 + bcs_flat.numel()] = bcs_flat

        lt_flat = leadtime.view(-1)
        lt_padded = torch.zeros(MAX_LT, dtype=dtype, device=device)
        lt_padded[0] = lt_flat.numel()
        lt_padded[1:1 + lt_flat.numel()] = lt_flat

        inp_flat = inp.contiguous().view(-1)

        header = torch.tensor([MAX_FL, MAX_BCS, MAX_LT, 0.0], dtype=dtype, device=device)

        packed = torch.cat([header, shape_f, fl_padded, bcs_padded, lt_padded, inp_flat])
        # Connect to dummy param for autograd
        packed = packed + self.dummy.sum() * 0

        return packed


class TurbtModelStage(nn.Module):
    """
    Stage 1: Runs the ORIGINAL turbt model's forward() method.

    This preserves exact correctness: same recursive multi-level hierarchy,
    same cross-level residual connections, same encode/decode paths, same
    ltimeMLP, local_att, sequence_factor_short/long, etc.

    The entire turbt model is contained in this stage. For memory savings,
    combine with ZeRO and the gradient checkpointing already implemented
    in turbt.py.
    """
    def __init__(self, turbt_model, params, tkhead_name):
        super().__init__()
        self.model = turbt_model
        self.params = params
        self.tkhead_name = tkhead_name
        self.nhlevels = params.hierarchical["nlevels"] if hasattr(params, "hierarchical") else 1

    def forward(self, packed):
        # Ensure the received tensor requires grad for backward pass
        if not packed.requires_grad:
            packed = packed.detach().requires_grad_(True)

        # Unpack fixed-size packed tensor from stage 0
        # Layout: [header(4) | shape_info(6) | fl_padded(MAX_FL) | bcs_padded(MAX_BCS) | lt_padded(MAX_LT) | inp_data(...)]
        header = packed[:4]
        MAX_FL = int(header[0].item())
        MAX_BCS = int(header[1].item())
        MAX_LT = int(header[2].item())

        offset = 4
        shape_info = packed[offset:offset + 6]
        offset += 6

        fl_padded = packed[offset:offset + MAX_FL]
        offset += MAX_FL
        n_labels = int(fl_padded[0].item())
        field_labels_1d = fl_padded[1:1 + n_labels].detach().to(torch.long)

        bcs_padded = packed[offset:offset + MAX_BCS]
        offset += MAX_BCS
        n_bcs = int(bcs_padded[0].item())
        bcs = bcs_padded[1:1 + n_bcs].detach()

        lt_padded = packed[offset:offset + MAX_LT]
        offset += MAX_LT
        n_lt = int(lt_padded[0].item())
        leadtime = lt_padded[1:1 + n_lt].detach()

        # inp_flat maintains grad connection to packed for backward
        inp_flat = packed[offset:]

        T = int(shape_info[0].item()); B = int(shape_info[1].item()); C = int(shape_info[2].item())
        D = int(shape_info[3].item()); H = int(shape_info[4].item()); W = int(shape_info[5].item())

        # Ensure model is on the same device as input data
        device = packed.device
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)

        data = inp_flat.view(T, B, C, D, H, W)
        state_labels = (field_labels_1d,)  # tuple of 1D tensors

        # Build ForwardOptionsBase matching what train.py does
        imod = self.nhlevels - 1  # start from finest level
        imod_bottom = determine_turt_levels(
            self.model.tokenizer_heads_params[self.tkhead_name][-1],
            data.shape[-3:], imod
        ) if imod > 0 else 0

        opts = ForwardOptionsBase(
            imod=imod,
            imod_bottom=imod_bottom,
            tkhead_name=self.tkhead_name,
            sequence_parallel_group=None,  # no SP in pipeline mode
            leadtime=leadtime,
            blockdict=None,  # no SP blocking
            cond_dict=None,
            cond_input=None,
            isgraph=False,
            field_labels_out=state_labels,
        )

        output = self.model(data, state_labels, bcs, opts)
        # output: [B, C_out, D, H, W] at finest level

        return output


# ============================================================================
# Pipeline creation
# ============================================================================

def create_matey_pipeline(base_model, params, global_rank=0):
    """
    Creates a 2-stage pipeline:
      Stage 0: TurbtPreprocessStage (data formatting, no parameters)
      Stage 1: TurbtModelStage (full turbt model with all levels)

    For more fine-grained pipeline parallelism with >2 stages, consider
    splitting the transformer blocks within TurbtModelStage using
    DeepSpeed's activation checkpoint boundaries.
    """
    tkhead_name = params.tokenizer_heads[0]["head_name"]
    nhlevels = params.hierarchical["nlevels"] if hasattr(params, "hierarchical") else 1

    layers = [
        TurbtPreprocessStage(nhlevels),
        TurbtModelStage(base_model, params, tkhead_name),
    ]

    if global_rank == 0:
        n_blocks_total = sum(
            len(list(base_model.module_blocks[str(i)]))
            for i in range(nhlevels)
        )
        print(f"Pipeline: 2 stages")
        print(f"  Stage 0: TurbtPreprocessStage (data prep)")
        print(f"  Stage 1: TurbtModelStage ({nhlevels} levels, "
              f"{n_blocks_total} transformer blocks total)")
        print(f"  Gradient checkpointing: "
              f"{getattr(params, 'gradient_checkpointing', False)}")

    return layers


# ============================================================================
# Validation
# ============================================================================

def validate_with_pipeline_matey(engine, params, global_rank, world_size,
                                  valid_dataset, pipeline_collate):
    """Validation loop for MATEY pipeline."""
    is_last = engine.is_last_stage()
    device = engine.device

    wrapped = MATEYPipelineDatasetWrapper(valid_dataset)
    valid_dl = DataLoader(
        wrapped, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
        collate_fn=pipeline_collate)

    NUM_VAL_STEPS = min(5, max(len(valid_dl), 1))
    total_loss = torch.tensor(0.0, device=device)
    valid_iter = iter(RepeatingLoader(valid_dl))

    engine.reset_activation_shape()

    for i in range(NUM_VAL_STEPS):
        try:
            loss = engine.eval_batch(data_iter=valid_iter)
        except Exception as e:
            import traceback
            print(f"[Rank {dist.get_rank()}] eval_batch failed step {i}: {e}",
                  flush=True)
            traceback.print_exc()
            return {'valid_loss': torch.tensor(float('nan'), device=device)}
        if is_last and loss is not None:
            total_loss += loss.detach()

    engine.reset_activation_shape()
    return {'valid_loss': total_loss / max(NUM_VAL_STEPS, 1)}


# ============================================================================
# Main training entry point
# ============================================================================

def train_with_pipeline_matey(params, global_rank, local_rank, world_size):
    """Main training function with DeepSpeed pipeline parallelism for MATEY."""

    # ---- data ----
    _, train_dataset, sampler = get_data_loader(
        params, params.train_data_paths,
        dist.is_initialized(), split='train',
        train_offset=params.embedding_offset,
        group_size=1, global_rank=global_rank, num_sp_groups=world_size)

    _, valid_dataset, _ = get_data_loader(
        params, params.valid_data_paths,
        dist.is_initialized(), split='val',
        group_size=1, global_rank=global_rank, num_sp_groups=world_size)

    # Auto-correct n_states (matches train.py lines 83-87)
    labels_total = [train_dataset.subset_dict[d] for d in train_dataset.subset_dict]
    labels_total = [i for sub in labels_total for i in sub]
    if params.n_states < max(labels_total) + 1:
        if global_rank == 0:
            print(f"Warning: n_states {params.n_states} too small, "
                  f"setting to {max(labels_total) + 1}")
        params.n_states = max(labels_total) + 1

    # ---- model ----
    if params.model_type == 'turbt':
        base_model = build_turbt(params)
    else:
        raise ValueError(
            f"Pipeline mode currently supports turbt only. "
            f"Use DDP for '{params.model_type}'.")

    if global_rank == 0:
        print("Creating MATEY pipeline (full-model wrapper)...")

    layers = create_matey_pipeline(base_model, params, global_rank)

    # ---- loss (matches DDP compute_loss_and_logs without accum_grad) ----
    def matey_loss_fn(outputs, targets):
        outputs = outputs.float()
        targets = targets.to(outputs.device)
        if targets.dim() == 2:
            res = outputs - targets
            return (res.pow(2).mean()) / (1e-7 + targets.pow(2).mean())
        spatial = tuple(range(outputs.ndim))[2:]
        res = outputs - targets
        raw = res.pow(2).mean(dim=spatial, keepdim=True) / \
              (1e-7 + targets.pow(2).mean(dim=spatial, keepdim=True))
        return raw.mean()

    # ---- DeepSpeed pipeline ----
    if not deepspeed.comm.is_initialized():
        deepspeed.comm.init_distributed(dist_backend='nccl')

    # With 2 layers, we need exactly 2 pipeline stages.
    # Override user's --pipeline_stages if it doesn't match.
    n_pipeline_stages = 2
    if params.pipeline_stages != n_pipeline_stages:
        if global_rank == 0:
            print(f"Note: overriding --pipeline_stages={params.pipeline_stages} "
                  f"to {n_pipeline_stages} (must match number of pipeline layers)")
        params.pipeline_stages = n_pipeline_stages

    pipeline_model = PipelineModule(
        layers=layers,
        loss_fn=matey_loss_fn,
        num_stages=n_pipeline_stages,
        partition_method='parameters',
    )

    dp_size = world_size // n_pipeline_stages
    # With pipeline mode (no SP), each rank gets full-resolution cubes.
    # Use micro_batch=1 to fit in GPU memory (model alone is ~45 GB on 64 GB GPUs).
    pp_micro_batch = 1
    # gradient_accumulation_steps must be >= num_stages for 1F1B schedule
    grad_accum = max(n_pipeline_stages, getattr(params, 'accum_grad', 2))
    ds_config = {
        "train_batch_size": pp_micro_batch * dp_size * grad_accum,
        "train_micro_batch_size_per_gpu": pp_micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": params.learning_rate,
                "weight_decay": getattr(params, 'weight_decay', 0.0),
                "torch_adam": True,
            },
        },
        "steps_per_print": 10,
        "pipeline": {
            "pipe_partitioned": False,
            "grad_partitioned": False,
        },
        "dataloader_drop_last": True,
    }

    if params.enable_amp:
        if torch.cuda.is_bf16_supported():
            ds_config["bf16"] = {"enabled": True}
        else:
            ds_config["fp16"] = {"enabled": True, "loss_scale": 0,
                                 "initial_scale_power": 16}

    if getattr(params, 'zero_stage', 0) > 0:
        ds_config["zero_optimization"] = {
            "stage": params.zero_stage,
            "reduce_bucket_size": 2.5e7,
        }

    engine, _, _, _ = deepspeed.initialize(
        model=pipeline_model, config=ds_config, training_data=None)

    if global_rank == 0:
        print(f"DeepSpeed engine initialized:")
        print(f"  Pipeline stages: {n_pipeline_stages}")
        print(f"  Data parallel size: {dp_size}")
        print(f"  Global batch size: {pp_micro_batch * dp_size * grad_accum}")
        print(f"  Micro batch per GPU: {pp_micro_batch}")
        print(f"  Gradient accumulation steps: {grad_accum}")

    # ---- dataloader ----
    pipeline_dataset = MATEYPipelineDatasetWrapper(train_dataset)

    def pipeline_collate(batch):
        input_tuples, targets = zip(*batch)
        batched_targets = torch.stack(targets, dim=0)
        batched_inputs = [torch.stack([t[i] for t in input_tuples], dim=0)
                          for i in range(len(input_tuples[0]))]
        return (tuple(batched_inputs), batched_targets)

    # In pipeline mode without sequence parallelism, each rank gets full-resolution
    # cubes (e.g., 128^3). The MultisetBatchSampler scales batch_size up to 8 for
    # small cubes, causing OOM (8 * 3 * 48 * 128^3 * 4 bytes ≈ 9.7 GB just for the
    # tokenizer input). Use a simple DistributedSampler with batch_size=1 instead.
    # Effective global batch = dp_size * 1 = 8, compensated by accum_grad.
    #
    # CRITICAL: DeepSpeed pipeline loads data on BOTH first and last stages.
    # First stage uses inputs, last stage uses targets. They must see the SAME
    # samples, so we shard by data-parallel rank (dp_size replicas), NOT world_size.
    # Ranks in the same pipeline (e.g., rank 0 and rank 8) get the same indices.
    pp_micro_batch = 1
    dp_rank = global_rank % dp_size  # data-parallel rank within pipeline group
    from torch.utils.data.distributed import DistributedSampler
    pp_sampler = DistributedSampler(
        pipeline_dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        drop_last=True,
    )
    pipeline_dl = DataLoader(
        pipeline_dataset,
        batch_size=pp_micro_batch,
        sampler=pp_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=pipeline_collate,
    )

    if global_rank == 0:
        print(f"Starting MATEY DeepSpeed Pipeline Training Loop...")

    train_iter = iter(RepeatingLoader(pipeline_dl))
    steps_per_epoch = len(pipeline_dl)
    total_steps = params.max_epochs * steps_per_epoch
    if global_rank == 0:
        print(f"Total training steps: {total_steps} ({params.max_epochs} epochs)")
        print(f"Dataloader: {len(pipeline_dl)} batches per epoch, dp_rank={dp_rank}")
        print(f"Pipeline stage: {engine.stage_id}, is_first={engine.is_first_stage()}, is_last={engine.is_last_stage()}")
    dist.barrier()
    if global_rank == 0:
        print(f"All ranks synchronized. Starting training loop...", flush=True)

    last_loss = torch.tensor(0.0, device=engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    t0 = time.time()
    epoch_t0 = time.time()

    for step in range(total_steps):
        if step < 3:
            print(f"[Rank {global_rank}] About to call engine.train_batch(), step={step}", flush=True)
        loss = engine.train_batch(data_iter=train_iter)
        if step < 3:
            print(f"[Rank {global_rank}] train_batch() returned, step={step}, loss={loss}", flush=True)

        if loss is not None:
            if engine.global_steps % ds_config['steps_per_print'] == 0:
                print(f"Step {engine.global_steps}/{total_steps}, "
                      f"Loss: {loss.item():.6f}")
            last_loss.copy_(loss.detach())

        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            if engine.is_last_stage():
                dt = time.time() - epoch_t0
                print(f"[Stage {engine.stage_id}] Epoch {epoch}/{params.max_epochs} "
                      f"({dt:.1f}s) | Train Loss: {last_loss.item():.6f}")

            dist.barrier()
            if global_rank == 0:
                print("  Starting validation...", flush=True)
            val = validate_with_pipeline_matey(
                engine, params, global_rank, world_size,
                valid_dataset, pipeline_collate)
            if global_rank == 0:
                print("  Validation completed.", flush=True)
            if engine.is_last_stage() and engine.mpu.get_data_parallel_rank() == 0:
                print(f"[Stage {engine.stage_id}] Epoch {epoch} "
                      f"| Valid Loss: {val['valid_loss'].item():.6f}")

            if (engine.is_last_stage()
                    and getattr(params, 'save_checkpoint', False)
                    and global_rank == 0):
                cp = os.path.join(params.experiment_dir,
                                  f'training_checkpoints/ckpt_epoch{epoch}')
                engine.save_checkpoint(cp)
                print(f"Checkpoint saved: {cp}")

            epoch_t0 = time.time()
            last_loss.zero_()

    total_dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(engine.device) / 1024**3
    if engine.is_last_stage() and engine.mpu.get_data_parallel_rank() == 0:
        print(f"\n--- Finished {total_steps} steps ({total_dt:.1f}s) ---")
        print(f"Peak GPU Memory: {peak_gb:.3f} GB")
        print("MATEY Pipeline training completed successfully!")
