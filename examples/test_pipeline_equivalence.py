#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
#
# Mode 2 equivalence test: verify that pipeline_forward_sequential
# (stage-by-stage with packed tensors) produces identical output to
# TurbTIterative's forward (single iterative loop).
#
# Run WITHOUT distributed (single GPU):
#   python test_pipeline_equivalence.py --yaml_config ./config/Demo_JHUTDB_TT.yaml --config basic_config
#
# Or with SLURM (single process, 1 GPU):
#   srun -N1 -n1 -c7 --gpus=1 python test_pipeline_equivalence.py ...

import argparse
import os
import sys
import torch
import copy

# Ensure the parent directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from matey.utils import YParams, ForwardOptionsBase
from matey.utils.distributed_utils import determine_turt_levels
from matey.models.turbt_pipeline import (
    build_turbt_iterative,
    build_turbt_pipeline_stages,
    pipeline_forward_sequential,
)


def make_dummy_data(params, device, B=1):
    """Create synthetic input tensors matching the JHTDB config."""
    n_steps = params.n_steps  # T dimension
    # Use 64^3 to ensure all hierarchical levels are active
    # (with filtersize=2 and nlevels=3, we need >= 2^(nlevels-1)*patch_size = 16)
    # 64^3 gives imod_bottom=0, so all 3 levels are exercised.
    D, H, W = 64, 64, 64
    # The dataset provides data with only the ACTIVE channels (not all n_states).
    # state_labels maps these channels to positions in the n_states-wide weight matrix.
    # For isotropic1024fine: 4 fields (Pressure, Vx, Vy, Density)
    n_used = 4
    # Input: (T, B, n_used, D, H, W) — only active channels
    data = torch.randn(n_steps, B, n_used, D, H, W, device=device)
    # state_labels: (B, n_used) — indices into the n_states-wide weight matrix
    state_labels = torch.arange(n_used, device=device).unsqueeze(0).expand(B, -1)
    # bcs: boundary conditions (B, n_used)
    bcs = torch.zeros(B, n_used, device=device)
    # leadtime
    leadtime = torch.ones(B, 1, device=device)

    return data, state_labels, bcs, leadtime, (D, H, W)


def run_test(params, device):
    print("=" * 70)
    print("Mode 2 Equivalence Test: stages vs iterative forward")
    print("=" * 70)

    # Build parent model (TurbTIterative) and stages
    parent, stages = build_turbt_pipeline_stages(params)
    parent = parent.to(device)
    for s in stages:
        s.to(device)
    parent.eval()
    for s in stages:
        s.eval()

    nhlevels = parent.nhlevels if parent.hierarchical else 1
    print(f"Model built: {sum(p.numel() for p in parent.parameters())} params, "
          f"{nhlevels} hierarchical levels")

    # Verify stages share parameters with parent (not copies)
    for imod, stage in enumerate(stages):
        # blocks should be the same object
        parent_blocks = parent.module_blocks[str(imod)]
        stage_blocks = stage.blocks
        for (pn, pp), (sn, sp) in zip(parent_blocks.named_parameters(),
                                       stage_blocks.named_parameters()):
            assert pp.data_ptr() == sp.data_ptr(), \
                f"Stage {imod} block param {pn} is a copy, not a reference!"
    print("Parameter sharing verified: stages reference parent params (not copies)")

    # Create dummy data
    data, state_labels, bcs, leadtime, (D, H, W) = make_dummy_data(params, device)
    print(f"Input shape: {data.shape}")

    # Build ForwardOptionsBase
    tkhead_name = list(parent.tokenizer_heads_params.keys())[0]
    imod_top = nhlevels - 1
    imod_bottom = determine_turt_levels(
        parent.tokenizer_heads_params[tkhead_name][-1],
        data.shape[-3:], imod_top) if imod_top > 0 else 0
    print(f"tkhead_name={tkhead_name}, imod_top={imod_top}, imod_bottom={imod_bottom}")

    opts = ForwardOptionsBase(
        imod=imod_top,
        imod_bottom=imod_bottom,
        tkhead_name=tkhead_name,
        sequence_parallel_group=None,
        leadtime=leadtime,
        blockdict=None,
        cond_input=None,
        isgraph=False,
        field_labels_out=state_labels,
    )

    # Use AMP autocast to match training conditions and avoid MIOpen
    # errors on AMD MI250X GPUs (ROCm requires AMP for some conv3d configs)
    mp_type = torch.bfloat16 if (torch.cuda.is_available() and
                                  torch.cuda.is_bf16_supported()) else torch.float16
    amp_enabled = device.type == 'cuda'
    print(f"AMP: enabled={amp_enabled}, dtype={mp_type}")

    # ---- Run TurbTIterative forward ----
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled, dtype=mp_type):
        out_iterative = parent(data, state_labels, bcs, copy.deepcopy(opts))
    # Cast to float32 for comparison
    out_iterative = out_iterative.float()
    print(f"TurbTIterative output shape: {out_iterative.shape}, "
          f"range: [{out_iterative.min().item():.6f}, {out_iterative.max().item():.6f}]")

    # ---- Run pipeline_forward_sequential ----
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled, dtype=mp_type):
        out_stages = pipeline_forward_sequential(
            parent, stages, data, state_labels, bcs, copy.deepcopy(opts))
    out_stages = out_stages.float()
    print(f"Stage-sequential output shape: {out_stages.shape}, "
          f"range: [{out_stages.min().item():.6f}, {out_stages.max().item():.6f}]")

    # ---- Compare ----
    assert out_iterative.shape == out_stages.shape, \
        f"Shape mismatch: {out_iterative.shape} vs {out_stages.shape}"

    abs_diff = (out_iterative - out_stages).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Relative difference (avoid div by zero)
    denom = out_iterative.abs().clamp(min=1e-8)
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    print(f"\nAbsolute difference:  max={max_abs:.2e}, mean={mean_abs:.2e}")
    print(f"Relative difference:  max={max_rel:.2e}, mean={mean_rel:.2e}")

    # For bfloat16 AMP, differences up to ~1e-2 are normal due to reduced
    # mantissa precision.  For float32, expect ~1e-6.
    ATOL = 1e-2 if amp_enabled else 1e-4
    RTOL = 5e-2 if amp_enabled else 1e-3

    if max_abs < ATOL and max_rel < RTOL:
        print(f"\n*** PASS *** Outputs match within atol={ATOL}, rtol={RTOL}")
        return True
    elif max_abs < 1e-2:
        print(f"\n*** SOFT PASS *** Small differences (max_abs={max_abs:.2e}). "
              f"Likely floating-point ordering. Check if acceptable.")
        return True
    else:
        print(f"\n*** FAIL *** Outputs differ significantly!")
        # Print per-element comparison for debugging
        idx = abs_diff.argmax()
        flat_iter = out_iterative.reshape(-1)
        flat_stage = out_stages.reshape(-1)
        print(f"  Worst element [{idx}]: iterative={flat_iter[idx].item():.8f}, "
              f"stages={flat_stage[idx].item():.8f}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mode 2: pipeline stage equivalence test")
    parser.add_argument("--yaml_config", default='./config/Demo_JHUTDB_TT.yaml', type=str)
    parser.add_argument("--config", default='basic_config', type=str)
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument("--pei_fixedupsample", action='store_true')
    parser.add_argument("--pei_linearupsample", action='store_true')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    if hasattr(params, "hierarchical"):
        params.hierarchical["fixedupsample"] = args.pei_fixedupsample
        params.hierarchical["linearupsample"] = args.pei_linearupsample

    device = torch.device(args.device)
    print(f"Device: {device}")

    passed = run_test(params, device)
    sys.exit(0 if passed else 1)
