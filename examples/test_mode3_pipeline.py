#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
#
# Mode 3 test: Multi-GPU pipeline parallel forward + backward.
#
# Tests PipelineParallelEngine with stages on separate GPUs within a
# single node.  Verifies that:
#   1. forward_single_device produces correct output across GPUs
#   2. Gradients flow back through all stages
#   3. Loss decreases over a few training steps
#
# Run: srun -N1 -n1 -c7 --gpus=3 python test_mode3_pipeline.py ...
# (Single process, 3 GPUs for 3 pipeline stages)

import argparse
import os
import sys
import torch
import torch.nn as nn
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from matey.utils import YParams, ForwardOptionsBase
from matey.utils.distributed_utils import determine_turt_levels
from matey.models.turbt_pipeline import (
    build_turbt_iterative,
    build_turbt_pipeline_stages,
    pipeline_forward_sequential,
    PipelineParallelEngine,
    set_pipeline_context,
)


def make_dummy_data(params, device, B=2):
    n_steps = params.n_steps
    D, H, W = 64, 64, 64
    n_used = 4
    data = torch.randn(n_steps, B, n_used, D, H, W, device=device)
    state_labels = torch.arange(n_used, device=device).unsqueeze(0).expand(B, -1)
    bcs = torch.zeros(B, n_used, device=device)
    leadtime = torch.ones(B, 1, device=device)
    target = torch.randn(B, n_used, D, H, W, device=device)
    return data, state_labels, bcs, leadtime, target


def test_forward_equivalence(params, devices):
    """Test 1: single-device sequential vs multi-device pipeline forward."""
    print("\n" + "=" * 60)
    print("Test 1: Forward equivalence (sequential vs multi-device)")
    print("=" * 60)

    parent, stages = build_turbt_pipeline_stages(params)
    parent = parent.to(devices[0])
    for s in stages:
        s.to(devices[0])  # start all on device 0
    parent.eval()
    for s in stages:
        s.eval()

    data, state_labels, bcs, leadtime, _ = make_dummy_data(params, devices[0])
    nhlevels = parent.nhlevels if parent.hierarchical else 1
    tkhead_name = list(parent.tokenizer_heads_params.keys())[0]
    imod_top = nhlevels - 1
    imod_bottom = determine_turt_levels(
        parent.tokenizer_heads_params[tkhead_name][-1],
        data.shape[-3:], imod_top) if imod_top > 0 else 0

    opts = ForwardOptionsBase(
        imod=imod_top, imod_bottom=imod_bottom, tkhead_name=tkhead_name,
        sequence_parallel_group=None, leadtime=leadtime, blockdict=None,
        cond_input=None, isgraph=False, field_labels_out=state_labels)

    # Reference: sequential on device 0
    with torch.no_grad():
        ref = pipeline_forward_sequential(parent, stages, data, state_labels, bcs, copy.deepcopy(opts))
    print(f"  Sequential output: shape={ref.shape}, range=[{ref.min():.4f}, {ref.max():.4f}]")

    # Now create pipeline engine with stages on different devices
    engine = PipelineParallelEngine(
        parent, stages,
        pp_group_ranks=[0],  # single-process, no dist
        devices=devices[:nhlevels],
        num_micro_batches=1,
    )
    engine.set_context(data=data, state_labels=state_labels, bcs=bcs, opts=copy.deepcopy(opts))

    with torch.no_grad():
        pp_out = engine.forward_single_device(data)
    print(f"  Pipeline output:   shape={pp_out.shape}, range=[{pp_out.min():.4f}, {pp_out.max():.4f}]")

    diff = (ref.to(devices[0]) - pp_out.to(devices[0])).abs()
    print(f"  Max abs diff: {diff.max().item():.2e}")

    if diff.max().item() < 1e-4:
        print("  *** PASS ***")
        return True
    else:
        print("  *** FAIL ***")
        return False


def test_gradient_flow(params, devices):
    """Test 2: verify gradients flow back through all stages."""
    print("\n" + "=" * 60)
    print("Test 2: Gradient flow through multi-device pipeline")
    print("=" * 60)

    parent, stages = build_turbt_pipeline_stages(params)
    nhlevels = parent.nhlevels if parent.hierarchical else 1

    engine = PipelineParallelEngine(
        parent, stages,
        pp_group_ranks=[0],
        devices=devices[:nhlevels],
        num_micro_batches=1,
    )

    data, state_labels, bcs, leadtime, target = make_dummy_data(params, devices[0])
    tkhead_name = list(parent.tokenizer_heads_params.keys())[0]
    imod_top = nhlevels - 1
    imod_bottom = determine_turt_levels(
        parent.tokenizer_heads_params[tkhead_name][-1],
        data.shape[-3:], imod_top) if imod_top > 0 else 0

    opts = ForwardOptionsBase(
        imod=imod_top, imod_bottom=imod_bottom, tkhead_name=tkhead_name,
        sequence_parallel_group=None, leadtime=leadtime, blockdict=None,
        cond_input=None, isgraph=False, field_labels_out=state_labels)

    engine.set_context(data=data, state_labels=state_labels, bcs=bcs, opts=opts)
    output = engine.forward_single_device(data)

    # Compute loss
    loss = F.mse_loss(output.to(devices[0]), target.to(devices[0]))
    loss.backward()

    # Check gradients exist on all stages
    all_have_grad = True
    for imod, stage in enumerate(stages):
        has_grad = False
        for name, p in stage.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        status = "OK" if has_grad else "NO GRAD"
        print(f"  Stage {imod} (device={devices[imod]}): {status}")
        if not has_grad:
            all_have_grad = False

    if all_have_grad:
        print("  *** PASS *** All stages received gradients")
    else:
        print("  *** FAIL *** Some stages have no gradients")
    return all_have_grad


def test_training_loop(params, devices, n_steps=5):
    """Test 3: mini training loop shows loss decreasing."""
    print("\n" + "=" * 60)
    print(f"Test 3: Training loop ({n_steps} steps)")
    print("=" * 60)

    parent, stages = build_turbt_pipeline_stages(params)
    nhlevels = parent.nhlevels if parent.hierarchical else 1

    engine = PipelineParallelEngine(
        parent, stages,
        pp_group_ranks=[0],
        devices=devices[:nhlevels],
        num_micro_batches=1,
    )

    # Use all parameters from parent for optimizer
    optimizer = torch.optim.Adam(parent.parameters(), lr=1e-3)

    data, state_labels, bcs, leadtime, target = make_dummy_data(params, devices[0], B=4)
    tkhead_name = list(parent.tokenizer_heads_params.keys())[0]
    imod_top = nhlevels - 1
    imod_bottom = determine_turt_levels(
        parent.tokenizer_heads_params[tkhead_name][-1],
        data.shape[-3:], imod_top) if imod_top > 0 else 0

    opts = ForwardOptionsBase(
        imod=imod_top, imod_bottom=imod_bottom, tkhead_name=tkhead_name,
        sequence_parallel_group=None, leadtime=leadtime, blockdict=None,
        cond_input=None, isgraph=False, field_labels_out=state_labels)

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        engine.set_context(data=data, state_labels=state_labels, bcs=bcs, opts=copy.deepcopy(opts))
        output = engine.forward_single_device(data)
        loss = F.mse_loss(output.to(devices[0]), target.to(devices[0]))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {step}: loss = {loss.item():.6f}")

    decreasing = losses[-1] < losses[0]
    if decreasing:
        print(f"  *** PASS *** Loss decreased: {losses[0]:.6f} -> {losses[-1]:.6f}")
    else:
        print(f"  *** FAIL *** Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}")
    return decreasing


def test_micro_batching(params, devices):
    """Test 4: verify micro-batching produces same result as full batch."""
    print("\n" + "=" * 60)
    print("Test 4: Micro-batching equivalence")
    print("=" * 60)

    parent, stages = build_turbt_pipeline_stages(params)
    nhlevels = parent.nhlevels if parent.hierarchical else 1
    parent.eval()
    for s in stages:
        s.eval()

    B = 4  # must be divisible by num_micro_batches
    data, state_labels, bcs, leadtime, _ = make_dummy_data(params, devices[0], B=B)
    tkhead_name = list(parent.tokenizer_heads_params.keys())[0]
    imod_top = nhlevels - 1
    imod_bottom = determine_turt_levels(
        parent.tokenizer_heads_params[tkhead_name][-1],
        data.shape[-3:], imod_top) if imod_top > 0 else 0

    opts = ForwardOptionsBase(
        imod=imod_top, imod_bottom=imod_bottom, tkhead_name=tkhead_name,
        sequence_parallel_group=None, leadtime=leadtime, blockdict=None,
        cond_input=None, isgraph=False, field_labels_out=state_labels)

    # Full batch (1 micro-batch)
    engine1 = PipelineParallelEngine(
        parent, stages, pp_group_ranks=[0],
        devices=devices[:nhlevels], num_micro_batches=1)
    engine1.set_context(data=data, state_labels=state_labels, bcs=bcs, opts=copy.deepcopy(opts))
    with torch.no_grad():
        out_full = engine1.forward_single_device(data)

    # 2 micro-batches
    engine2 = PipelineParallelEngine(
        parent, stages, pp_group_ranks=[0],
        devices=devices[:nhlevels], num_micro_batches=2)
    engine2.set_context(data=data, state_labels=state_labels, bcs=bcs, opts=copy.deepcopy(opts))
    with torch.no_grad():
        out_mb = engine2.forward_micro_batched(data)

    diff = (out_full.to(devices[0]) - out_mb.to(devices[0])).abs()
    print(f"  Full batch output:  shape={out_full.shape}")
    print(f"  Micro-batch output: shape={out_mb.shape}")
    print(f"  Max abs diff: {diff.max().item():.2e}")

    if diff.max().item() < 1e-4:
        print("  *** PASS ***")
        return True
    else:
        print("  *** FAIL ***")
        return False


if __name__ == '__main__':
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description="Mode 3: pipeline parallel tests")
    parser.add_argument("--yaml_config", default='./config/Demo_JHUTDB_TT.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--pei_fixedupsample", action='store_true')
    parser.add_argument("--pei_linearupsample", action='store_true')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if hasattr(params, "hierarchical"):
        params.hierarchical["fixedupsample"] = args.pei_fixedupsample
        params.hierarchical["linearupsample"] = args.pei_linearupsample

    n_gpus = torch.cuda.device_count()
    nhlevels = params.hierarchical["nlevels"] if hasattr(params, "hierarchical") else 1
    print(f"Available GPUs: {n_gpus}, Pipeline stages needed: {nhlevels}")

    if n_gpus < nhlevels:
        print(f"WARNING: Only {n_gpus} GPUs available, need {nhlevels} for multi-device test.")
        print("Falling back to single-device (all stages on cuda:0)")
        devices = [torch.device('cuda:0')] * nhlevels
    else:
        devices = [torch.device(f'cuda:{i}') for i in range(nhlevels)]

    print(f"Device assignment: {devices}")
    print()

    results = {}
    results['forward_equiv'] = test_forward_equivalence(params, devices)
    results['gradient_flow'] = test_gradient_flow(params, devices)
    results['training_loop'] = test_training_loop(params, devices)
    results['micro_batching'] = test_micro_batching(params, devices)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
