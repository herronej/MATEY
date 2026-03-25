#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
#
# Mode 3 training: distributed pipeline-parallel with real data.
#
# Uses the same dataloader, optimizer, scheduler, and loss as the
# standard Trainer, but replaces the model forward with
# pipeline_forward_distributed (differentiable inter-stage communication).
#
# Produces JSON loss logs in the same format as train.py's Trainer,
# so you can directly compare loss curves against the DDP baseline.
#
# Launch (same as submit_JHTDB_demo.sh but with pipeline stages):
#   srun -N2 -n$((2*8)) -c7 --gpu-bind=closest python train_pipeline.py \
#       --yaml_config ./config/Demo_JHUTDB_TT.yaml --config basic_config

import argparse
import os
import sys
import json
import socket
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.nn.functional as F
import copy
from datetime import timedelta
from einops import rearrange
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from matey.utils import (YParams, ForwardOptionsBase, setup_dist, Timer,
                         log_to_file, log_versions)
from matey.utils.distributed_utils import (
    get_sequence_parallel_group, locate_group, add_weight_decay,
    CosineNoIncrease, determine_turt_levels, parse_slurm_nodelist)
from matey.data_utils.datasets import get_data_loader
from matey.models.turbt_pipeline import (
    build_turbt_pipeline_stages,
    pipeline_forward_distributed,
    prefilter_all_levels,
)
from matey.utils.training_utils import compute_loss_and_logs, update_loss_logs_inplace_eval


def run(args):
    # ---- Setup ----
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params.use_ddp = True   # needed for setup_dist
    params.use_fsdp = False
    if hasattr(params, "hierarchical"):
        params.hierarchical["fixedupsample"] = args.pei_fixedupsample
        params.hierarchical["linearupsample"] = args.pei_linearupsample

    device, world_size, local_rank, global_rank = setup_dist(params)
    print(f"Rank {global_rank}/{world_size}, local={local_rank}, "
          f"host={socket.gethostname()}", flush=True)

    nhlevels = params.hierarchical["nlevels"] if hasattr(params, "hierarchical") else 1
    num_pp_groups = world_size // nhlevels
    assert world_size == num_pp_groups * nhlevels, \
        f"world_size ({world_size}) must be divisible by nhlevels ({nhlevels}). " \
        f"Use -n{num_pp_groups * nhlevels} in srun."
    my_pp_group_id = global_rank // nhlevels
    my_stage_idx = global_rank % nhlevels
    pp_ranks = list(range(my_pp_group_id * nhlevels,
                          my_pp_group_id * nhlevels + nhlevels))

    if global_rank == 0:
        print(f"Pipeline config: {nhlevels} stages, {num_pp_groups} groups", flush=True)

    # ---- Create link groups (all ranks must participate in new_group) ----
    all_link_groups = {}
    for pg in range(num_pp_groups):
        base = pg * nhlevels
        for li in range(nhlevels - 1):
            ranks_in_link = [base + li, base + li + 1]
            grp = dist.new_group(ranks_in_link, timeout=timedelta(minutes=10))
            all_link_groups[(pg, li)] = grp

    # Also create a DDP group for gradient sync across pipeline groups
    # (all ranks at the same stage position form a DDP group)
    stage_ddp_groups = {}
    for si in range(nhlevels):
        ddp_ranks = [pg * nhlevels + si for pg in range(num_pp_groups)]
        grp = dist.new_group(ddp_ranks, timeout=timedelta(minutes=10))
        stage_ddp_groups[si] = grp

    my_link_groups = [all_link_groups[(my_pp_group_id, li)] for li in range(nhlevels - 1)]
    my_ddp_group = stage_ddp_groups[my_stage_idx]

    # ---- Data (must be loaded before model to determine n_states) ----
    # Within each PP group, all ranks at different stages need the same data.
    # The sampler uses my_pp_group_id as rank, num_pp_groups as num_replicas.
    params['batch_size'] = int(params.batch_size // num_pp_groups)

    train_loader, train_dataset, train_sampler = get_data_loader(
        params, params.train_data_paths, dist.is_initialized(), split='train',
        train_offset=params.embedding_offset,
        group_size=1, global_rank=my_pp_group_id, num_sp_groups=num_pp_groups)
    valid_loader, valid_dataset, val_sampler = get_data_loader(
        params, params.valid_data_paths, dist.is_initialized(), split='val',
        group_size=1, global_rank=my_pp_group_id, num_sp_groups=num_pp_groups)

    # Match Trainer's n_states correction: bump up if dataset labels exceed config
    labels_total = [train_dataset.subset_dict[d] for d in train_dataset.subset_dict]
    labels_total = [item for sublist in labels_total for item in sublist]
    if params.n_states < max(labels_total) + 1:
        if global_rank == 0:
            print(f"Warning: n_states {params.n_states} too small for max label "
                  f"{max(labels_total)}, setting to {max(labels_total)+1}", flush=True)
        params.n_states = max(labels_total) + 1

    # Broadcast corrected n_states to all ranks
    n_states_t = torch.tensor([params.n_states], dtype=torch.long, device=device)
    dist.broadcast(n_states_t, src=0)
    params.n_states = int(n_states_t.item())

    # ---- Build model (after n_states is finalized) ----
    # Build on CPU first, then only move this rank's stage to GPU.
    # This avoids OOM from having all stages on every GPU.
    torch.manual_seed(42)  # same init across all ranks
    parent, stages = build_turbt_pipeline_stages(params)
    my_stage = stages[my_stage_idx]

    # Move parent to CPU first (frees GPU memory for unused stages),
    # then move only this rank's stage to GPU.  Since stages hold references
    # to parent sub-modules, my_stage.to(device) moves exactly the modules
    # this stage needs to GPU while the rest stay on CPU.
    parent.cpu()
    my_stage.to(device)

    # Count params in this stage
    stage_params = sum(p.numel() for p in my_stage.parameters())
    if global_rank == 0:
        total_params = sum(p.numel() for p in parent.parameters())
        print(f"Model: {total_params} total params, n_states={params.n_states}", flush=True)
    print(f"Rank {global_rank} (stage {my_stage_idx}): {stage_params} params on GPU", flush=True)

    # ---- Optimizer only for this stage's parameters ----
    optimizer = optim.AdamW(my_stage.parameters(), lr=params.learning_rate,
                            weight_decay=params.weight_decay)
    mp_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.half
    gscaler = amp.GradScaler(enabled=(mp_type == torch.half and params.enable_amp))

    # Cosine scheduler with warmup
    k = params.warmup_steps
    sched_epochs = params.max_epochs
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.01, end_factor=1.0, total_iters=k)
    decay = CosineNoIncrease(optimizer, eta_min=params.learning_rate / 100,
                             T_max=sched_epochs * params.epoch_size // params.accum_grad - k)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, decay], [k])

    # ---- Output dir ----
    expDir = os.path.join(params.exp_dir, args.config, args.run_name)
    if global_rank == 0:
        os.makedirs(expDir, exist_ok=True)
        os.makedirs(os.path.join(expDir, 'training_checkpoints'), exist_ok=True)
        log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        print(f"Output dir: {expDir}", flush=True)

    timer = Timer(enable_sync=False)

    # ---- Training loop ----
    best_valid_loss = 1e6
    # Determine which rank in each PP group is the tail (computes loss)
    tail_stage = nhlevels - 1
    # For logging: the tail rank of PP group 0 is the one that prints losses
    tail_rank_group0 = tail_stage  # rank index within the group

    for epoch in range(params.max_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        parent.train()
        for s in stages:
            s.train()

        data_iter = iter(train_loader)
        num_batches = min(len(train_loader), params.epoch_size)
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        epoch_start = timer.get_time()

        for batch_idx in range(num_batches):
            data = next(data_iter)
            inp, dset_index, field_labels, bcs, tar, leadtime = map(
                lambda x: x.to(device),
                [data[v] for v in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
            inp = rearrange(inp, 'b t c d h w -> t b c d h w')
            dset_type = train_dataset.sub_dsets[dset_index[0]].type
            tkhead_name = train_dataset.sub_dsets[dset_index[0]].tkhead_name
            blockdict = getattr(train_dataset.sub_dsets[dset_index[0]], "blockdict", None)

            imod = nhlevels - 1
            imod_bottom = determine_turt_levels(
                parent.tokenizer_heads_params[tkhead_name][-1],
                inp.shape[-3:], imod) if imod > 0 else 0

            opts = ForwardOptionsBase(
                imod=imod, imod_bottom=imod_bottom,
                tkhead_name=tkhead_name,
                sequence_parallel_group=None,
                leadtime=leadtime,
                blockdict=copy.deepcopy(blockdict),
                cond_input=None, isgraph=False,
                field_labels_out=field_labels)

            with amp.autocast(params.enable_amp, dtype=mp_type):
                output = pipeline_forward_distributed(
                    my_stage, my_stage_idx, nhlevels, device,
                    my_link_groups, pp_ranks, imod_bottom,
                    inp, field_labels, bcs, opts, parent)

                if my_stage_idx == tail_stage and output is not None:
                    tar = tar.to(device)
                    spatial_dims = tuple(range(2, output.ndim))
                    raw_loss = (output - tar).pow(2).mean(spatial_dims) / \
                               (1e-7 + tar.pow(2).mean(spatial_dims))
                    loss = raw_loss.mean() / params.accum_grad
                    log_nrmse = raw_loss.sqrt().mean().item()
                    train_losses.append(log_nrmse)
                else:
                    loss = None
                    log_nrmse = 0.0

            # Backward
            if loss is not None:
                gscaler.scale(loss).backward()

            # Gradient accumulation + step
            if (1 + batch_idx) % params.accum_grad == 0:
                # Sync gradients across pipeline groups (DDP-style all-reduce per stage)
                for p in my_stage.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=my_ddp_group)

                gscaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(my_stage.parameters(), 0.5)
                gscaler.step(optimizer)
                gscaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # Per-batch logging: tail rank of PP group 0 prints
            if my_stage_idx == tail_stage and my_pp_group_id == 0 and batch_idx % params.log_interval == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Train Loss {log_nrmse:.6f} "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}", flush=True)

        # ---- Epoch train metrics ----
        if my_stage_idx == nhlevels - 1 and len(train_losses) > 0:
            avg_train = sum(train_losses) / len(train_losses)
        else:
            avg_train = 0.0
        avg_train_t = torch.tensor([avg_train], device=device)
        dist.all_reduce(avg_train_t, op=dist.ReduceOp.SUM)
        avg_train_t = avg_train_t / num_pp_groups  # average across pipeline groups

        dist.barrier()  # train epoch barrier
        train_time = timer.get_time() - epoch_start

        # ---- Validation ----
        parent.eval()
        for s in stages:
            s.eval()
        valid_losses = []
        valid_iter = iter(valid_loader)
        num_valid = min(len(valid_loader), params.epoch_size)

        with torch.no_grad():
            for idx in range(num_valid):
                data = next(valid_iter)
                inp, dset_index, field_labels, bcs, tar, leadtime = map(
                    lambda x: x.to(device),
                    [data[v] for v in ["input", "dset_idx", "field_labels", "bcs", "label", "leadtime"]])
                inp = rearrange(inp, 'b t c d h w -> t b c d h w')
                tkhead_name = valid_dataset.sub_dsets[dset_index[0]].tkhead_name
                blockdict = getattr(valid_dataset.sub_dsets[dset_index[0]], "blockdict", None)

                imod = nhlevels - 1
                imod_bottom = determine_turt_levels(
                    parent.tokenizer_heads_params[tkhead_name][-1],
                    inp.shape[-3:], imod) if imod > 0 else 0

                opts = ForwardOptionsBase(
                    imod=imod, imod_bottom=imod_bottom,
                    tkhead_name=tkhead_name,
                    sequence_parallel_group=None,
                    leadtime=leadtime,
                    blockdict=copy.deepcopy(blockdict),
                    cond_input=None, isgraph=False,
                    field_labels_out=field_labels)

                with amp.autocast(params.enable_amp, dtype=mp_type):
                    output = pipeline_forward_distributed(
                        my_stage, my_stage_idx, nhlevels, device,
                        my_link_groups, pp_ranks, imod_bottom,
                        inp, field_labels, bcs, opts, parent)

                if my_stage_idx == nhlevels - 1 and output is not None:
                    tar = tar.to(device)
                    spatial_dims = tuple(range(2, output.ndim))
                    raw = (output - tar).pow(2).mean(spatial_dims) / \
                          (1e-7 + tar.pow(2).mean(spatial_dims))
                    valid_losses.append(raw.sqrt().mean().item())

        if my_stage_idx == nhlevels - 1 and len(valid_losses) > 0:
            avg_valid = sum(valid_losses) / len(valid_losses)
        else:
            avg_valid = 0.0
        avg_valid_t = torch.tensor([avg_valid], device=device)
        dist.all_reduce(avg_valid_t, op=dist.ReduceOp.SUM)
        avg_valid_t = avg_valid_t / num_pp_groups

        dist.barrier()  # valid epoch barrier
        valid_time = timer.get_time() - epoch_start - train_time

        # ---- Logging + checkpointing ----
        # Collect updated stage params back to parent (CPU) for saving
        # Each rank's stage has its own updated params; we gather to rank 0.
        # For simplicity, each rank saves its own stage, and rank 0 assembles.
        stage_state = my_stage.state_dict()
        # Move stage state to CPU for saving
        stage_state_cpu = {k: v.cpu() for k, v in stage_state.items()}

        if global_rank == 0:
            train_nrmse = avg_train_t.item()
            valid_nrmse = avg_valid_t.item()
            logs_out = {
                'epoch': epoch + 1,
                'train_nrmse': train_nrmse,
                'valid_nrmse': valid_nrmse,
                'lr': optimizer.param_groups[0]['lr'],
                'time/train': train_time,
                'time/valid': valid_time,
            }
            with open(os.path.join(expDir, f'train_log_epoch{epoch}.json'), 'w') as fp:
                json.dump(logs_out, fp)
            print(f"Epoch {epoch+1}: train_nrmse={train_nrmse:.6f}, "
                  f"valid_nrmse={valid_nrmse:.6f}, "
                  f"train_time={train_time:.1f}s, valid_time={valid_time:.1f}s", flush=True)

        # Save per-stage checkpoints (each rank saves its own stage)
        ckpt_dir = os.path.join(expDir, 'training_checkpoints')
        torch.save({
            'epoch': epoch + 1,
            'stage_idx': my_stage_idx,
            'stage_state': stage_state_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(ckpt_dir, f'ckpt_stage{my_stage_idx}_group{my_pp_group_id}.tar'))

        if global_rank == 0 and valid_nrmse < best_valid_loss:
            best_valid_loss = valid_nrmse
            print(f"  Best valid_nrmse={valid_nrmse:.6f}", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    if global_rank == 0:
        print("DONE", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default='pp_train', type=str)
    parser.add_argument("--yaml_config", default='./config/Demo_JHUTDB_TT.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--pei_fixedupsample", action='store_true')
    parser.add_argument("--pei_linearupsample", action='store_true')
    args = parser.parse_args()
    run(args)
