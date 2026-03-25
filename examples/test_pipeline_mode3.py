#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
#
# Mode 3: Distributed pipeline-parallel training across multiple GPUs/nodes.
#
# Design:
#   - Each rank owns ONE pipeline stage.
#   - Ranks [0..nhlevels-1] form pipeline group 0, [nhlevels..2*nhlevels-1]
#     form group 1, etc.
#   - Inter-stage activation transfer uses broadcast within 2-rank subgroups
#     (NCCL-safe; avoids point-to-point send/recv issues).
#   - Every rank pre-filters data locally and only runs its own stage.
#   - Backward only produces gradients for the tail rank's stage params
#     (full 1F1B gradient pipelining is left as future work).
#
# Launch across 2 Frontier nodes:
#   srun -N2 -n$((2*8)) -c7 --gpu-bind=closest python test_pipeline_mode3.py ...

import argparse
import os
import sys
import socket
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
import copy
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from matey.utils import YParams, ForwardOptionsBase
from matey.utils.distributed_utils import determine_turt_levels, parse_slurm_nodelist
from matey.models.turbt_pipeline import (
    build_turbt_pipeline_stages,
    prefilter_all_levels,
    pack_interstage,
    unpack_interstage,
)


def setup_dist():
    world_size = int(os.environ.get('SLURM_NTASKS', '1'))
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    local_rank = int(os.environ.get('SLURM_LOCALID', '0'))
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'
    if os.getenv("SLURM_STEP_NODELIST") is not None:
        os.environ['MASTER_ADDR'] = parse_slurm_nodelist(os.environ["SLURM_STEP_NODELIST"])[0]
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '3442'
    dist.init_process_group(backend="nccl", init_method='env://',
                            rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=10))
    torch.cuda.set_device(local_rank)
    return torch.device(local_rank), world_size, local_rank, rank


def make_dummy_data(params, device, B=4):
    n_steps = params.n_steps
    D, H, W = 64, 64, 64
    n_used = 4
    data = torch.randn(n_steps, B, n_used, D, H, W, device=device)
    state_labels = torch.arange(n_used, device=device).unsqueeze(0).expand(B, -1)
    bcs = torch.zeros(B, n_used, device=device)
    leadtime = torch.ones(B, 1, device=device)
    target = torch.randn(B, n_used, D, H, W, device=device)
    return data, state_labels, bcs, leadtime, target


def broadcast_packed_tensor(packed, src_rank, group, device, max_size=50_000_000):
    """Broadcast a variable-length packed tensor within a process group.

    NCCL-friendly: uses allreduce-style broadcast (all ranks in group must call).

    Protocol:
      1. Broadcast the size (1 element) from src_rank.
      2. Allocate buffer on all ranks.
      3. Broadcast the data from src_rank.
    """
    size_t = torch.tensor([packed.numel() if dist.get_rank() == src_rank else 0],
                          dtype=torch.long, device=device)
    dist.broadcast(size_t, src=src_rank, group=group)
    n = int(size_t.item())

    if dist.get_rank() == src_rank:
        buf = packed.contiguous()
    else:
        buf = torch.zeros(n, dtype=packed.dtype, device=device)
    dist.broadcast(buf, src=src_rank, group=group)
    return buf


def run_pipeline_forward(my_stage, my_stage_idx, nhlevels, device,
                         link_groups, link_src_ranks, pp_ranks, imod_bottom,
                         data, state_labels, bcs, opts, parent):
    """Run one forward pass through the pipeline.

    Communication uses broadcast within 2-rank subgroups (NCCL-safe).
    Each link_group[i] connects stage i to stage i+1.
    """
    # Pre-filter and set context locally
    filtered, blockdicts = prefilter_all_levels(parent, data, opts.blockdict)
    n_filt = parent.nhlevels - 1 - my_stage_idx
    my_stage.imod_bottom = imod_bottom
    my_stage.set_context(
        state_labels=state_labels, bcs=bcs,
        tkhead_name=opts.tkhead_name,
        blockdict=copy.deepcopy(blockdicts[n_filt]),
        filtered_data=filtered[n_filt],
        leadtime=opts.leadtime, cond_input=opts.cond_input,
        sequence_parallel_group=opts.sequence_parallel_group)

    # Stage 0 receives nothing; stages 1+ receive from previous
    if my_stage_idx == imod_bottom:
        # Create sentinel
        packed_in = pack_interstage(
            None,
            torch.zeros(1, device=device, dtype=data.dtype),
            torch.ones(1, device=device, dtype=data.dtype))
    else:
        # Participate in broadcast from previous stage
        link_idx = my_stage_idx - 1
        prev_rank = pp_ranks[my_stage_idx - 1]
        packed_in = broadcast_packed_tensor(
            torch.empty(0, device=device, dtype=data.dtype),  # placeholder
            src_rank=prev_rank,
            group=link_groups[link_idx],
            device=device)

    # Run my stage
    with amp.autocast(enabled=True, dtype=torch.bfloat16):
        packed_out = my_stage(packed_in)

    # Send to next stage (if not tail)
    if my_stage_idx < nhlevels - 1:
        link_idx = my_stage_idx
        my_rank = pp_ranks[my_stage_idx]
        packed_out_bc = broadcast_packed_tensor(
            packed_out, src_rank=my_rank,
            group=link_groups[link_idx],
            device=device)

    # If not the tail, also participate in any link_groups we're the receiver of
    # (already handled above)

    if my_stage_idx == nhlevels - 1:
        x_pred, _, _, _ = unpack_interstage(packed_out)
        return x_pred
    else:
        return None


def run_test(params, device, world_size, local_rank, global_rank):
    torch.manual_seed(42)
    parent, stages = build_turbt_pipeline_stages(params)
    nhlevels = parent.nhlevels if parent.hierarchical else 1
    parent = parent.to(device)
    for s in stages:
        s.to(device)

    if global_rank == 0:
        print("=" * 70)
        print(f"Mode 3: Distributed Pipeline Parallel")
        print(f"  World size: {world_size}, stages: {nhlevels}")
        print(f"  Params: {sum(p.numel() for p in parent.parameters())}")
        print("=" * 70, flush=True)

    # Assign pipeline groups
    num_pp_groups = world_size // nhlevels
    my_pp_group_id = global_rank // nhlevels
    my_stage_idx = global_rank % nhlevels
    is_active = my_pp_group_id < num_pp_groups and global_rank < num_pp_groups * nhlevels

    pp_ranks = list(range(my_pp_group_id * nhlevels,
                          my_pp_group_id * nhlevels + nhlevels)) if is_active else []

    print(f"Rank {global_rank}: {'active' if is_active else 'idle'}, "
          f"PP group {my_pp_group_id}, stage {my_stage_idx}, "
          f"host={socket.gethostname()}", flush=True)

    # Create link groups: one per adjacent stage pair within each PP group.
    # link_groups[i] connects stage i and stage i+1 within my PP group.
    # All ranks must participate in new_group, so we create groups for ALL
    # pipeline groups at once.
    all_link_groups = {}  # (pp_group_id, link_idx) -> group
    for pg in range(num_pp_groups):
        base = pg * nhlevels
        for link_idx in range(nhlevels - 1):
            ranks_in_link = [base + link_idx, base + link_idx + 1]
            grp = dist.new_group(ranks_in_link, timeout=timedelta(minutes=5))
            all_link_groups[(pg, link_idx)] = grp

    if is_active:
        my_link_groups = [all_link_groups[(my_pp_group_id, li)]
                          for li in range(nhlevels - 1)]
        my_link_src_ranks = [pp_ranks[li] for li in range(nhlevels - 1)]
        my_stage = stages[my_stage_idx]
    else:
        my_link_groups = []
        my_link_src_ranks = []
        my_stage = None

    # Sync before starting tests
    dist.barrier()

    if not is_active:
        # Idle ranks must match the barrier count of active ranks:
        # 1 (after test 1) + 1 (after test 2) + 3 (inside training loop) + 1 (final) = 6
        for _ in range(6):
            dist.barrier()
        return True

    # Create data (same seed per PP group)
    torch.manual_seed(100 + my_pp_group_id)
    data, state_labels, bcs, leadtime, target = make_dummy_data(params, device, B=4)

    tkhead_name = list(parent.tokenizer_heads_params.keys())[0]
    imod_top = nhlevels - 1
    imod_bottom = determine_turt_levels(
        parent.tokenizer_heads_params[tkhead_name][-1],
        data.shape[-3:], imod_top) if imod_top > 0 else 0

    opts = ForwardOptionsBase(
        imod=imod_top, imod_bottom=imod_bottom,
        tkhead_name=tkhead_name, sequence_parallel_group=None,
        leadtime=leadtime, blockdict=None, cond_input=None,
        isgraph=False, field_labels_out=state_labels)

    # ---- Test 1: Forward ----
    if global_rank == 0:
        print(f"\n--- Test 1: Distributed forward ---", flush=True)

    with torch.no_grad():
        output = run_pipeline_forward(
            my_stage, my_stage_idx, nhlevels, device,
            my_link_groups, my_link_src_ranks, pp_ranks, imod_bottom,
            data, state_labels, bcs, opts, parent)

    if my_stage_idx == nhlevels - 1:
        print(f"  Rank {global_rank} (tail, group {my_pp_group_id}): "
              f"output shape={output.shape}, "
              f"range=[{output.min().item():.4f}, {output.max().item():.4f}]", flush=True)

    dist.barrier()

    # ---- Test 2: Forward + backward (tail only) ----
    if global_rank == 0:
        print(f"\n--- Test 2: Forward + backward ---", flush=True)

    output = run_pipeline_forward(
        my_stage, my_stage_idx, nhlevels, device,
        my_link_groups, my_link_src_ranks, pp_ranks, imod_bottom,
        data, state_labels, bcs, opts, parent)

    if my_stage_idx == nhlevels - 1 and output is not None:
        spatial_dims = tuple(range(2, output.ndim))
        loss = ((output - target).pow(2).mean(spatial_dims) /
                (1e-7 + target.pow(2).mean(spatial_dims))).mean()
        loss.backward()
        grad_count = sum(1 for p in parent.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
        total_p = sum(1 for _ in parent.parameters())
        print(f"  Rank {global_rank} (tail, group {my_pp_group_id}): "
              f"loss={loss.item():.6f}, grads={grad_count}/{total_p}", flush=True)

    dist.barrier()

    # ---- Test 3: Training steps (tail rank updates) ----
    if global_rank == 0:
        print(f"\n--- Test 3: Training (3 steps) ---", flush=True)

    optimizer = torch.optim.AdamW(parent.parameters(), lr=1e-3)

    for step in range(3):
        optimizer.zero_grad()
        torch.manual_seed(200 + step + my_pp_group_id * 1000)
        data_s, sl_s, bcs_s, lt_s, tgt_s = make_dummy_data(params, device, B=4)
        opts_s = ForwardOptionsBase(
            imod=imod_top, imod_bottom=imod_bottom,
            tkhead_name=tkhead_name, sequence_parallel_group=None,
            leadtime=lt_s, blockdict=None, cond_input=None,
            isgraph=False, field_labels_out=sl_s)

        output = run_pipeline_forward(
            my_stage, my_stage_idx, nhlevels, device,
            my_link_groups, my_link_src_ranks, pp_ranks, imod_bottom,
            data_s, sl_s, bcs_s, opts_s, parent)

        if my_stage_idx == nhlevels - 1 and output is not None:
            spatial_dims = tuple(range(2, output.ndim))
            loss = ((output - tgt_s).pow(2).mean(spatial_dims) /
                    (1e-7 + tgt_s.pow(2).mean(spatial_dims))).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parent.parameters(), 0.5)
            optimizer.step()
            if my_pp_group_id == 0:
                print(f"  Step {step}: loss={loss.item():.6f} (group 0 tail)", flush=True)

        dist.barrier()

    dist.barrier()
    if global_rank == 0:
        print(f"\n{'=' * 70}")
        print("Mode 3 tests completed successfully.")
        print(f"{'=' * 70}", flush=True)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/Demo_JHUTDB_TT.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--pei_fixedupsample", action='store_true')
    parser.add_argument("--pei_linearupsample", action='store_true')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if hasattr(params, "hierarchical"):
        params.hierarchical["fixedupsample"] = args.pei_fixedupsample
        params.hierarchical["linearupsample"] = args.pei_linearupsample

    device, world_size, local_rank, global_rank = setup_dist()
    print(f"Rank {global_rank}/{world_size}, local={local_rank}, "
          f"host={socket.gethostname()}", flush=True)

    try:
        passed = run_test(params, device, world_size, local_rank, global_rank)
    except Exception as e:
        print(f"Rank {global_rank}: EXCEPTION: {e}", flush=True)
        import traceback
        traceback.print_exc()
        passed = False

    dist.destroy_process_group()
    sys.exit(0 if passed else 1)
