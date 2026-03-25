# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
#
# Pipeline-parallel TurbT: Modes 1, 2, and 3.
#
# Mode 1: TurbTIterative — drop-in single-device iterative forward.
# Mode 2: pipeline_forward_sequential — stage-by-stage, one device.
# Mode 3: PipelineParallelEngine — GPipe-style manual PP with micro-batching,
#          multi-device stage placement, and DDP replication across groups.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from einops import rearrange, repeat
from .turbt import TurbT, build_turbt
from ..data_utils.shared_utils import normalize_spatiotemporal_persample
from ..data_utils.utils import construct_filterkernel, construct_filterkernel2D
from ..utils.forward_options import ForwardOptionsBase
import copy
from operator import mul
from functools import reduce


# ===================================================================
#  Standalone helpers (no self, no cross-device references)
# ===================================================================

def _preembed(x, state_labels, space_bag):
    x = rearrange(x, 't b c d h w -> t b d h w c')
    x = space_bag(x, state_labels)
    x = rearrange(x, 't b d h w c -> t b c d h w')
    return x

def _embed_tokens(x_pre, tokenizer):
    T = x_pre.shape[0]
    x = rearrange(x_pre, 't b c d h w -> (t b) c d h w')
    x = tokenizer[-1](x)
    x = rearrange(x, '(t b) c d h w -> t b c d h w', t=T)
    return x

def _get_t_pos_area(x_pre, embed_ensemble, blockdict=None):
    T, B = x_pre.shape[:2]
    space_dims = x_pre.shape[3:]
    expand_patterns = {0: "d -> b t d h w", 1: "h -> b t d h w", 2: "w -> b t d h w"}
    ps = embed_ensemble[-1].patch_size
    ntokendim, delta = [], []
    for idim, dim in enumerate(space_dims):
        ntokendim.append(dim // ps[idim])
        delta.append(1.0 / dim * ps[idim])
    t_pos_area = torch.zeros(B, T, ntokendim[0], ntokendim[1], ntokendim[2],
                             2 + len(space_dims), device=x_pre.device)
    t_pos_area[:, :, :, :, :, 0] = repeat(
        torch.arange(T, device=x_pre.device), "t -> b t d h w",
        b=B, d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
    for idim, dim in enumerate(space_dims):
        dim_seq = repeat(
            torch.arange(delta[idim] * 0.5, 1.0, delta[idim], device=x_pre.device),
            expand_patterns[idim], b=B, t=T, d=ntokendim[0],
            h=ntokendim[1], w=ntokendim[2])
        t_pos_area[:, :, :, :, :, idim + 1] = dim_seq
    t_pos_area[:, :, :, :, :, -1] = reduce(mul, delta)
    if blockdict is not None:
        zxy_start = blockdict["zxy_start"]
        Lzxy = blockdict["Lzxy"]
        for idim in range(len(space_dims)):
            t_pos_area[..., idim + 1] = t_pos_area[..., idim + 1] * Lzxy[idim] + zxy_start[idim]
        t_pos_area[..., -1] = t_pos_area[..., -1] * reduce(mul, Lzxy)
    return t_pos_area

def _patchsequence_simple(x, state_labels, space_bag, embed_ensemble, blockdict):
    x_pre = _preembed(x, state_labels, space_bag)
    x_tok = _embed_tokens(x_pre, embed_ensemble)
    x_tok = rearrange(x_tok, 't b c d h w -> t b c (d h w)')
    t_pos_area = _get_t_pos_area(x_pre, embed_ensemble, blockdict=blockdict)
    t_pos_area = rearrange(t_pos_area, 'b t d h w c -> b t (d h w) c')
    return x_tok, t_pos_area

def _decode_simple(x_padding, space_dims, embed_ensemble, debed_ensemble):
    T, B = x_padding.shape[:2]
    ps_c = embed_ensemble[-1].patch_size
    ntokendim = [dim // ps_c[idim] for idim, dim in enumerate(space_dims)]
    ntoken_coarse = reduce(mul, ntokendim)
    x_coarsen = x_padding[:, :, :, :ntoken_coarse]
    x_coarsen = rearrange(x_coarsen, 't b c (d h w) -> (t b) c d h w',
                          d=ntokendim[0], h=ntokendim[1], w=ntokendim[2])
    x_coarsen = debed_ensemble[-1](x_coarsen)
    x_coarsen = rearrange(x_coarsen, '(t b) c d h w -> t b c d h w', t=T)
    return x_coarsen

def _sequence_factor_short(x, embed_ensemble, tkhead_name, tspace_dims, nfact=2):
    B, C, TL = x.shape
    ps_c = embed_ensemble[-1].patch_size
    ntokendim = [dim // ps_c[idim] for idim, dim in enumerate(tspace_dims[1:])]
    d, h, w = ntokendim
    if h // nfact < 4:
        nfact = max(1, h // 4)
    if nfact < 2:
        return x, nfact
    nfactd = 1 if d == 1 else nfact
    x = rearrange(x, 'b c (t d h w) -> b c t d h w', t=tspace_dims[0], d=d, h=h, w=w)
    x = x.unfold(3, d // nfactd, d // nfactd).unfold(4, h // nfact, h // nfact).unfold(5, w // nfact, w // nfact)
    x = rearrange(x, 'b c t nd nh nw d h w -> (b nd nh nw) c (t d h w)')
    return x, nfact

def _sequence_factor_long(x, embed_ensemble, tkhead_name, tspace_dims, nfact=2):
    if nfact < 2:
        return x
    B, C, TL = x.shape
    ps_c = embed_ensemble[-1].patch_size
    ntokendim = [dim // ps_c[idim] for idim, dim in enumerate(tspace_dims[1:])]
    d, h, w = ntokendim
    nfactd = 1 if d == 1 else nfact
    x = rearrange(x, '(b nd nh nw) c (t d h w) -> b c t nd nh nw d h w',
                  b=B // (nfactd * nfact * nfact), nd=nfactd, nh=nfact, nw=nfact,
                  d=d // nfactd, h=h // nfact, w=w // nfact)
    x = rearrange(x, 'b c t nd nh nw d h w -> b c t (nd d) (nh h) (nw w)')
    x = rearrange(x, 'b c t d h w -> b c (t d h w)')
    return x

def _filterdata(data, kernel_3d, kernel_2d, filtersize, blockdict=None):
    assert data.ndim == 6
    with torch.no_grad():
        T, B, C, D, H, W = data.shape
        data_flat = rearrange(data, 't b c d h w -> (t b c) d h w')
        if D == 1:
            filtered = F.conv3d(data_flat[:, None, :, :, :],
                                kernel_2d.to(data.device), stride=(1, filtersize, filtersize))
        else:
            filtered = F.conv3d(data_flat[:, None, :, :, :],
                                kernel_3d.to(data.device), stride=filtersize)
        filtered = rearrange(filtered, '(t b c) c1 d h w -> t b (c c1) d h w', t=T, b=B, c=C)
        if blockdict is not None:
            if D == 1:
                blockdict["Ind_dim"] = [D, H // filtersize, W // filtersize]
            else:
                blockdict["Ind_dim"] = [D // filtersize, H // filtersize, W // filtersize]
    return filtered, blockdict


# ===================================================================
#  Packed-tensor protocol
# ===================================================================

_META_LEN = 8

def pack_interstage(x_pred, data_mean, data_std, transfer_dtype=None):
    """Pack stage output for inter-stage transfer.
    
    Args:
        transfer_dtype: if set, cast the packed tensor to this dtype to reduce
                        transfer size. Default None = keep original dtype.
    """
    if x_pred is None:
        mean_flat = data_mean.reshape(-1).detach()
        std_flat = data_std.reshape(-1).detach()
        meta = torch.zeros(_META_LEN, dtype=mean_flat.dtype, device=mean_flat.device)
        meta[5] = mean_flat.numel()
        meta[6] = std_flat.numel()
        result = torch.cat([meta, mean_flat, std_flat])
        return result.to(transfer_dtype) if transfer_dtype is not None else result
    B, C, D, H, W = x_pred.shape
    x_flat = x_pred.reshape(-1)
    mean_flat = data_mean.reshape(-1)
    std_flat = data_std.reshape(-1)
    meta = torch.tensor([B, C, D, H, W, mean_flat.numel(), std_flat.numel(), 1.0],
                        dtype=x_pred.dtype, device=x_pred.device)
    result = torch.cat([meta, x_flat, mean_flat, std_flat])
    return result.to(transfer_dtype) if transfer_dtype is not None else result

def unpack_interstage(flat):
    meta = flat[:_META_LEN]
    has_pred = meta[7].item() > 0.5
    n_mean, n_std = int(meta[5].item()), int(meta[6].item())
    offset = _META_LEN
    if has_pred:
        B, C, D, H, W = (int(meta[i].item()) for i in range(5))
        n = B * C * D * H * W
        x_pred = flat[offset:offset + n].reshape(B, C, D, H, W)
        offset += n
    else:
        x_pred = None
    mean_flat = flat[offset:offset + n_mean]; offset += n_mean
    std_flat = flat[offset:offset + n_std]
    shape = (int(meta[0]), int(meta[1]), int(meta[2]),
             int(meta[3]), int(meta[4])) if has_pred else None
    return x_pred, mean_flat, std_flat, shape


# ===================================================================
#  TurbTStage: one hierarchical level, Tensor -> Tensor
# ===================================================================

class TurbTStage(nn.Module):
    def __init__(self, parent: TurbT, imod: int,
                 is_first: bool = False, is_last: bool = False,
                 imod_bottom: int = 0):
        super().__init__()
        self.imod = imod
        self.is_first = is_first
        self.is_last = is_last
        self.imod_bottom = imod_bottom

        self.blocks = parent.module_blocks[str(imod)]
        self.upscale_space = (parent.module_upscale_space[str(imod)]
                              if str(imod) in parent.module_upscale_space else None)
        self.upscale_space2D = (parent.module_upscale_space2D[str(imod)]
                                if str(imod) in parent.module_upscale_space2D else None)
        self.space_bag = parent.space_bag[imod]
        self.embed_ensemble = parent.tokenizer_ensemble_heads[imod]
        self.posbias = parent.posbias[imod]
        self.ltimeMLP = (parent.ltimeMLP[imod]
                         if parent.leadtime and imod < len(parent.ltimeMLP) else None)
        self.inconMLP = (parent.inconMLP[imod]
                         if parent.cond_input and hasattr(parent, 'inconMLP')
                         and imod < len(parent.inconMLP) else None)

        self.filtersize = getattr(parent, 'filtersize', None)
        self.nhlevels = getattr(parent, 'nhlevels', 1)
        self.embed_dim = parent.embed_dim
        self._has_leadtime = parent.leadtime
        self._has_cond_input = parent.cond_input

        if parent.datafilter_kernel is not None:
            self.register_buffer('filt3d', parent.datafilter_kernel.clone(), persistent=False)
        else:
            self.filt3d = None
        if getattr(parent, 'datafilter_kernel2D', None) is not None:
            self.register_buffer('filt2d', parent.datafilter_kernel2D.clone(), persistent=False)
        else:
            self.filt2d = None

        self._ctx = {}

    def set_context(self, *, state_labels, bcs, tkhead_name, blockdict,
                    filtered_data, leadtime=None, cond_input=None,
                    sequence_parallel_group=None):
        self._ctx = dict(state_labels=state_labels, bcs=bcs,
                         tkhead_name=tkhead_name, blockdict=blockdict,
                         filtered_data=filtered_data, leadtime=leadtime,
                         cond_input=cond_input, seq_group=sequence_parallel_group)

    def _upsample(self, data):
        B, C, D, H, W = data.shape
        if D == 1 and self.upscale_space2D is not None:
            return self.upscale_space2D(data)
        elif self.upscale_space is not None:
            return self.upscale_space(data)
        return data

    def forward(self, packed_input: torch.Tensor) -> torch.Tensor:
        x_pred, _, _, _ = unpack_interstage(packed_input)
        ctx = self._ctx
        state_labels = ctx['state_labels']
        bcs = ctx['bcs']
        tkhead_name = ctx['tkhead_name']
        blockdict = copy.deepcopy(ctx['blockdict'])
        x = ctx['filtered_data']
        seq_group = ctx['seq_group']
        imod = self.imod
        T, B, _, D, H, W = x.shape
        x, data_mean, data_std = normalize_spatiotemporal_persample(x)

        leadtime = ctx['leadtime']
        if self._has_leadtime and leadtime is not None and self.ltimeMLP is not None:
            leadtime = self.ltimeMLP(leadtime)
        else:
            leadtime = None
        if self._has_cond_input and ctx['cond_input'] is not None and self.inconMLP is not None:
            lt_cond = self.inconMLP(ctx['cond_input'])
            leadtime = lt_cond if leadtime is None else leadtime + lt_cond

        embed_ens = self.embed_ensemble[tkhead_name]["embed"]
        debed_ens = self.embed_ensemble[tkhead_name]["debed"]
        x_tok, t_pos_area = _patchsequence_simple(
            x, state_labels, self.space_bag, embed_ens, blockdict)
        x_enc = rearrange(x_tok, 't b c ntoken_tot -> b c (t ntoken_tot)')

        if self.posbias is not None and t_pos_area is not None:
            pb = self.posbias(t_pos_area, mask_padding=None, use_zpos=(D > 1))
            pb = rearrange(pb, 'b t L c -> b c (t L)')
            x_enc = x_enc + pb; del pb

        local_att = imod > self.imod_bottom
        nfact = 1
        if local_att:
            nfact = (max(2 ** (2 * (imod - self.imod_bottom)) //
                         blockdict["nproc_blocks"][-1], 1)
                     if blockdict is not None
                     else max(2 ** (2 * (imod - self.imod_bottom)), 1))
            x_enc, nfact = _sequence_factor_short(
                x_enc, embed_ens, tkhead_name, [T, D, H, W], nfact=nfact)

        for iblk, blk in enumerate(self.blocks):
            b_mod = x_enc.shape[0]
            lt_blk = leadtime.repeat(b_mod // B, 1) if leadtime is not None else None
            x_enc = blk(x_enc, sequence_parallel_group=seq_group, bcs=bcs,
                        leadtime=lt_blk if iblk == 0 else None,
                        mask_padding=None, local_att=local_att)

        if local_att:
            x_enc = _sequence_factor_long(
                x_enc, embed_ens, tkhead_name, [T, D, H, W], nfact=nfact)

        x_dec = rearrange(x_enc, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)
        x_dec = _decode_simple(x_dec, [D, H, W], embed_ens, debed_ens)

        x_correct = x_dec[-1]; del x_dec
        if x_pred is not None:
            x_filt, _ = _filterdata(x_correct[None, ...], self.filt3d, self.filt2d, self.filtersize)
            filtered_eps = self._upsample(x_filt[-1])
            x_correct = x_correct - filtered_eps
            x_pred_up = self._upsample(x_pred)
            x_correct = x_correct + x_pred_up

        if self.is_last:
            x_correct = x_correct[:, state_labels[0], ...] * data_std[-1] + data_mean[-1]

        return pack_interstage(x_correct, data_mean, data_std, transfer_dtype=torch.bfloat16)


# ===================================================================
#  TurbTIterative: single-device iterative forward (Mode 1)
# ===================================================================

class TurbTIterative(TurbT):
    def forward(self, data, state_labels, bcs, opts: ForwardOptionsBase):
        imod_top = opts.imod
        imod_bottom = opts.imod_bottom
        tkhead_name = opts.tkhead_name
        sequence_parallel_group = opts.sequence_parallel_group
        leadtime_orig = opts.leadtime
        blockdict_orig = opts.blockdict
        cond_input = opts.cond_input
        isgraph = opts.isgraph
        field_labels_out = opts.field_labels_out
        if opts.refine_ratio is not None:
            raise ValueError("Adaptive tokenization not set up in TurbT")
        if field_labels_out is None:
            field_labels_out = state_labels
        if isgraph:
            return super().forward(data, state_labels, bcs, opts)
        filtered_levels = [data]
        blockdict_levels = [copy.deepcopy(blockdict_orig)]
        for _ in range(self.nhlevels - 1):
            filt, bd = self.filterdata(filtered_levels[-1],
                                       blockdict=copy.deepcopy(blockdict_levels[-1]))
            filtered_levels.append(filt)
            blockdict_levels.append(bd)
        x_pred = None
        for imod in range(imod_bottom, imod_top + 1):
            n_filt = self.nhlevels - 1 - imod
            x = filtered_levels[n_filt]
            blockdict = copy.deepcopy(blockdict_levels[n_filt])
            T, B, _, D, H, W = x.shape
            x, data_mean, data_std = normalize_spatiotemporal_persample(x)
            leadtime = leadtime_orig
            if self.leadtime and leadtime is not None:
                leadtime = self.ltimeMLP[imod](leadtime)
            else:
                leadtime = None
            if self.cond_input and cond_input is not None:
                leadtime = (self.inconMLP[imod](cond_input) if leadtime is None
                            else leadtime + self.inconMLP[imod](cond_input))
            x, patch_ids, patch_ids_ref, mask_padding, _, _, tpa, _ = \
                self.get_patchsequence(x, state_labels, tkhead_name,
                                       refineind=None, blockdict=blockdict,
                                       ilevel=imod, isgraph=False)
            x = rearrange(x, 't b c ntoken_tot -> b c (t ntoken_tot)')
            if self.posbias[imod] is not None and tpa is not None:
                pb = self.posbias[imod](tpa, mask_padding=mask_padding, use_zpos=(D > 1))
                pb = rearrange(pb, 'b t L c -> b c (t L)')
                x = x + pb; del pb
            mask4att = None if (mask_padding is not None and mask_padding.all()) else mask_padding
            local_att = imod > imod_bottom
            nfact = 1
            if local_att:
                nfact = (max(2 ** (2 * (imod - imod_bottom)) //
                             blockdict["nproc_blocks"][-1], 1)
                         if blockdict is not None
                         else max(2 ** (2 * (imod - imod_bottom)), 1))
                x, nfact = self.sequence_factor_short(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)
            for iblk, blk in enumerate(self.module_blocks[str(imod)]):
                b_mod = x.shape[0]
                lt_blk = leadtime.repeat(b_mod // B, 1) if leadtime is not None else None
                x = blk(x, sequence_parallel_group=sequence_parallel_group, bcs=bcs,
                        leadtime=lt_blk if iblk == 0 else None,
                        mask_padding=mask4att, local_att=local_att)
            if local_att:
                x = self.sequence_factor_long(x, imod, tkhead_name, [T, D, H, W], nfact=nfact)
            x = rearrange(x, 'b c (t ntoken_tot) -> t b c ntoken_tot', t=T)
            x = self.get_spatiotemporalfromsequence(
                x, patch_ids, patch_ids_ref, [D, H, W], tkhead_name, ilevel=imod, isgraph=False)
            x_correct = x[-1]; del x
            if x_pred is not None:
                x_filter = self.filterdata(x_correct[None, ...])[0][-1]
                filtered_eps = self.upsampeldata(x_filter, imod)
                x_correct = x_correct - filtered_eps
                x_pred = self.upsampeldata(x_pred, imod)
                x_correct = x_correct + x_pred
            x_pred = x_correct
        return x_correct[:, state_labels[0], ...] * data_std[-1] + data_mean[-1]


# ===================================================================
#  Builders
# ===================================================================

def build_turbt_iterative(params):
    return TurbTIterative(
        tokenizer_heads=params.tokenizer_heads, embed_dim=params.embed_dim,
        num_heads=params.num_heads, processor_blocks=params.processor_blocks,
        n_states=params.n_states,
        sts_model=getattr(params, 'sts_model', False),
        sts_train=getattr(params, 'sts_train', False),
        leadtime=hasattr(params, "leadtime_max") and params.leadtime_max >= 0,
        cond_input=getattr(params, 'supportdata', False),
        n_steps=params.n_steps, bias_type=params.bias_type,
        replace_patch=getattr(params, 'replace_patch', True),
        hierarchical=getattr(params, 'hierarchical', None),
        notransposed=getattr(params, 'notransposed', False),
    )

def build_turbt_pipeline_stages(params):
    parent = build_turbt_iterative(params)
    nhlevels = parent.nhlevels if parent.hierarchical else 1
    stages = []
    for imod in range(nhlevels):
        stages.append(TurbTStage(parent, imod,
                                  is_first=(imod == 0),
                                  is_last=(imod == nhlevels - 1)))
    return parent, stages

get_turbt_pipeline_stages = build_turbt_pipeline_stages


# ===================================================================
#  Context injection
# ===================================================================

def prefilter_all_levels(parent, data, blockdict_orig):
    filtered = [data]
    blockdicts = [copy.deepcopy(blockdict_orig)]
    for _ in range(parent.nhlevels - 1):
        filt, bd = parent.filterdata(filtered[-1],
                                     blockdict=copy.deepcopy(blockdicts[-1]))
        filtered.append(filt)
        blockdicts.append(bd)
    return filtered, blockdicts

def set_pipeline_context(stages, *, parent, data, state_labels, bcs, opts):
    filtered, blockdicts = prefilter_all_levels(parent, data, opts.blockdict)
    for imod, stage in enumerate(stages):
        n_filt = parent.nhlevels - 1 - imod
        stage.imod_bottom = opts.imod_bottom
        stage.set_context(
            state_labels=state_labels, bcs=bcs,
            tkhead_name=opts.tkhead_name,
            blockdict=copy.deepcopy(blockdicts[n_filt]),
            filtered_data=filtered[n_filt],
            leadtime=opts.leadtime, cond_input=opts.cond_input,
            sequence_parallel_group=opts.sequence_parallel_group)


# ===================================================================
#  Mode 2: Sequential pipeline forward (single device)
# ===================================================================

def pipeline_forward_sequential(parent, stages, data, state_labels,
                                bcs, opts: ForwardOptionsBase):
    set_pipeline_context(stages, parent=parent, data=data,
                         state_labels=state_labels, bcs=bcs, opts=opts)
    dummy_m = torch.zeros(1, device=data.device, dtype=data.dtype)
    dummy_s = torch.ones(1, device=data.device, dtype=data.dtype)
    packed = pack_interstage(None, dummy_m, dummy_s)
    for imod in range(opts.imod_bottom, opts.imod + 1):
        packed = stages[imod](packed)
    x_pred, _, _, _ = unpack_interstage(packed)
    return x_pred


# ===================================================================
#  Mode 3: GPipe-style manual pipeline parallel engine
# ===================================================================

class PipelineParallelEngine:
    """Manual GPipe-style pipeline parallelism for TurbT.

    Architecture:
        - Each stage lives on a dedicated device (GPU).
        - Activations (packed tensors) are sent between stages via
          point-to-point torch.distributed operations.
        - Multiple pipeline groups can run in parallel (hybrid PP+DDP).
        - Micro-batching splits the batch for pipeline fill/drain overlap.

    Usage:
        engine = PipelineParallelEngine(
            parent, stages, pp_group_ranks=[0,1,2],
            devices=[torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2')],
            num_micro_batches=4,
        )

        # In training loop:
        engine.set_context(data=inp, state_labels=field_labels, bcs=bcs, opts=opts)
        output = engine.forward(data)           # pipelined forward
        loss = loss_fn(output, target)
        loss.backward()                         # gradients flow back through stages
        optimizer.step()
    """

    def __init__(self, parent, stages, pp_group_ranks, devices,
                 num_micro_batches=1):
        """
        Args:
            parent: TurbTIterative that owns all parameters.
            stages: list[TurbTStage], one per hierarchical level.
            pp_group_ranks: list of global ranks forming this pipeline group
                            (e.g., [0, 1, 2] for a 3-stage pipeline).
            devices: list of torch.device, one per stage.
            num_micro_batches: number of micro-batches for GPipe scheduling.
        """
        self.parent = parent
        self.stages = stages
        self.pp_group_ranks = pp_group_ranks
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.num_stages = len(stages)

        assert len(devices) == len(stages), \
            f"Need one device per stage: {len(devices)} devices, {len(stages)} stages"

        # Move each stage to its device
        for stage, device in zip(self.stages, self.devices):
            stage.to(device)

        # Determine this rank's role
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        if self.global_rank in pp_group_ranks:
            self.stage_idx = pp_group_ranks.index(self.global_rank)
            self.my_stage = stages[self.stage_idx]
            self.my_device = devices[self.stage_idx]
        else:
            self.stage_idx = -1
            self.my_stage = None
            self.my_device = None

        # Create pipeline process group
        if dist.is_initialized() and len(pp_group_ranks) > 1:
            self.pp_group = dist.new_group(pp_group_ranks)
        else:
            self.pp_group = None

    def set_context(self, *, data, state_labels, bcs, opts):
        """Pre-filter and inject context into all stages."""
        # Pre-filter on the device that has the data
        filtered, blockdicts = prefilter_all_levels(self.parent, data, opts.blockdict)
        for imod, stage in enumerate(self.stages):
            n_filt = self.parent.nhlevels - 1 - imod
            stage.imod_bottom = opts.imod_bottom
            # Move filtered data to the stage's device
            filt_data = filtered[n_filt].to(self.devices[imod])
            stage.set_context(
                state_labels=state_labels.to(self.devices[imod]),
                bcs=bcs.to(self.devices[imod]),
                tkhead_name=opts.tkhead_name,
                blockdict=copy.deepcopy(blockdicts[n_filt]),
                filtered_data=filt_data,
                leadtime=opts.leadtime.to(self.devices[imod]) if opts.leadtime is not None else None,
                cond_input=opts.cond_input.to(self.devices[imod]) if opts.cond_input is not None else None,
                sequence_parallel_group=opts.sequence_parallel_group)

    def forward_single_device(self, data):
        """Run all stages sequentially on their respective devices.

        No micro-batching, no distributed communication — just device placement.
        Useful for single-node multi-GPU pipeline parallelism.
        Gradients flow back through .to() calls via autograd.
        """
        imod_bottom = self.stages[0].imod_bottom
        imod_top = len(self.stages) - 1

        dummy_m = torch.zeros(1, device=self.devices[0], dtype=data.dtype)
        dummy_s = torch.ones(1, device=self.devices[0], dtype=data.dtype)
        packed = pack_interstage(None, dummy_m, dummy_s)

        for imod in range(imod_bottom, imod_top + 1):
            stage = self.stages[imod]
            device = self.devices[imod]
            packed = packed.to(device)
            packed = stage(packed)

        # Move result back to first device for loss computation
        packed = packed.to(self.devices[0])
        x_pred, _, _, _ = unpack_interstage(packed)
        return x_pred

    def forward_micro_batched(self, data):
        """GPipe-style micro-batched pipeline forward.

        Splits the batch dimension of the context's filtered_data and
        runs micro-batches through the stages. This overlaps computation
        when combined with backward passes (full GPipe schedule).

        For now, this implements the simple "all forwards then all backwards"
        GPipe schedule on a single process that owns all stages.
        """
        M = self.num_micro_batches
        imod_bottom = self.stages[0].imod_bottom
        imod_top = len(self.stages) - 1

        # Split context data into micro-batches along batch dim
        orig_ctxs = [copy.copy(s._ctx) for s in self.stages]

        # Collect outputs from all micro-batches
        outputs = []

        for mb in range(M):
            # Slice the context for this micro-batch
            for imod, stage in enumerate(self.stages):
                ctx = orig_ctxs[imod]
                B = ctx['filtered_data'].shape[1]
                mb_size = B // M
                start = mb * mb_size
                end = start + mb_size if mb < M - 1 else B

                stage._ctx = dict(
                    state_labels=ctx['state_labels'][start:end],
                    bcs=ctx['bcs'][start:end],
                    tkhead_name=ctx['tkhead_name'],
                    blockdict=copy.deepcopy(ctx['blockdict']),
                    filtered_data=ctx['filtered_data'][:, start:end],
                    leadtime=ctx['leadtime'][start:end] if ctx['leadtime'] is not None else None,
                    cond_input=ctx['cond_input'][start:end] if ctx['cond_input'] is not None else None,
                    seq_group=ctx['seq_group'],
                )

            # Run this micro-batch through all stages
            dummy_m = torch.zeros(1, device=self.devices[0], dtype=data.dtype)
            dummy_s = torch.ones(1, device=self.devices[0], dtype=data.dtype)
            packed = pack_interstage(None, dummy_m, dummy_s)

            for imod in range(imod_bottom, imod_top + 1):
                packed = packed.to(self.devices[imod])
                packed = self.stages[imod](packed)

            packed = packed.to(self.devices[0])
            x_pred_mb, _, _, _ = unpack_interstage(packed)
            outputs.append(x_pred_mb)

        # Restore original contexts
        for imod, stage in enumerate(self.stages):
            stage._ctx = orig_ctxs[imod]

        # Concatenate micro-batch outputs along batch dim
        return torch.cat(outputs, dim=0)


# ===================================================================
#  Loss function for pipeline
# ===================================================================

class PipelineLoss(nn.Module):
    def __init__(self, accum_grad=1):
        super().__init__()
        self.accum_grad = accum_grad

    def forward(self, packed_output, target):
        x_pred, _, _, _ = unpack_interstage(packed_output)
        spatial_dims = tuple(range(x_pred.ndim))[2:]
        raw_loss = (x_pred - target).pow(2).mean(spatial_dims) / \
                   (1e-7 + target.pow(2).mean(spatial_dims))
        return raw_loss.mean() / self.accum_grad


# ===================================================================
#  DeepSpeed PipelineModule (for reference / future use)
# ===================================================================

def build_deepspeed_pipeline(params, **ds_kwargs):
    try:
        from deepspeed.pipe import PipelineModule
    except ImportError:
        raise ImportError("DeepSpeed required. pip install deepspeed")
    parent, stages = build_turbt_pipeline_stages(params)
    pipe_model = PipelineModule(layers=stages, **ds_kwargs)
    return pipe_model, {'parent': parent, 'stages': stages}


# ===================================================================
#  Mode 3 distributed: differentiable inter-stage communication
# ===================================================================

# Maximum elements per broadcast chunk. NCCL can struggle with very large
# single broadcasts (>100M elements) especially when many groups operate
# concurrently.  16M elements × 2 bytes (bf16) = 32 MB per chunk.
_BROADCAST_CHUNK = 16 * 1024 * 1024  # 16M elements


def _chunked_broadcast(tensor, src, group):
    """Broadcast a tensor in chunks to avoid NCCL buffer exhaustion."""
    if tensor.numel() <= _BROADCAST_CHUNK:
        dist.broadcast(tensor, src=src, group=group)
        return
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), _BROADCAST_CHUNK):
        end = min(start + _BROADCAST_CHUNK, flat.numel())
        dist.broadcast(flat[start:end], src=src, group=group)


class PipelineSendRecv(torch.autograd.Function):
    """Differentiable broadcast for pipeline-parallel activation transfer.

    Forward:  broadcast packed activation from sender to receiver (chunked).
    Backward: broadcast gradient from receiver back to sender (chunked).

    Both operations use dist.broadcast within a 2-rank process group,
    which is NCCL-safe.  Large tensors are split into chunks to avoid
    NCCL buffer exhaustion under concurrent multi-group traffic.
    """
    @staticmethod
    def forward(ctx, packed_tensor, src_rank, link_group, device):
        ctx.src_rank = src_rank
        ctx.link_group = link_group
        ctx.device = device
        ctx.my_rank = dist.get_rank()

        # Broadcast size
        size_t = torch.tensor(
            [packed_tensor.numel() if ctx.my_rank == src_rank else 0],
            dtype=torch.long, device=device)
        dist.broadcast(size_t, src=src_rank, group=link_group)
        n = int(size_t.item())

        # Broadcast dtype (as int8 code)
        dtype_code = torch.tensor(
            [_dtype_to_code(packed_tensor.dtype) if ctx.my_rank == src_rank else 0],
            dtype=torch.long, device=device)
        dist.broadcast(dtype_code, src=src_rank, group=link_group)
        dtype = _code_to_dtype(int(dtype_code.item()))

        # Allocate / prepare buffer
        if ctx.my_rank == src_rank:
            buf = packed_tensor.contiguous()
        else:
            buf = torch.zeros(n, dtype=dtype, device=device)

        # Chunked broadcast
        _chunked_broadcast(buf, src=src_rank, group=link_group)
        return buf

    @staticmethod
    def backward(ctx, grad_output):
        link_group = ctx.link_group
        src_rank = ctx.src_rank
        device = ctx.device
        my_rank = ctx.my_rank

        ranks = dist.get_process_group_ranks(link_group)
        recv_rank = [r for r in ranks if r != src_rank][0]

        # Broadcast size from receiver
        size_t = torch.tensor(
            [grad_output.numel() if my_rank == recv_rank else 0],
            dtype=torch.long, device=device)
        dist.broadcast(size_t, src=recv_rank, group=link_group)
        n = int(size_t.item())

        # Broadcast dtype
        dtype_code = torch.tensor(
            [_dtype_to_code(grad_output.dtype) if my_rank == recv_rank else 0],
            dtype=torch.long, device=device)
        dist.broadcast(dtype_code, src=recv_rank, group=link_group)
        dtype = _code_to_dtype(int(dtype_code.item()))

        if my_rank == recv_rank:
            grad_buf = grad_output.contiguous()
        else:
            grad_buf = torch.zeros(n, dtype=dtype, device=device)

        _chunked_broadcast(grad_buf, src=recv_rank, group=link_group)
        return grad_buf, None, None, None


# Dtype encoding for broadcast
_DTYPE_MAP = {
    torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
    torch.float64: 3, torch.int64: 4, torch.int32: 5,
}
_DTYPE_RMAP = {v: k for k, v in _DTYPE_MAP.items()}

def _dtype_to_code(dtype):
    return _DTYPE_MAP.get(dtype, 0)

def _code_to_dtype(code):
    return _DTYPE_RMAP.get(code, torch.float32)


def pipeline_send_recv(packed, src_rank, link_group, device):
    """Differentiable inter-stage activation transfer."""
    return PipelineSendRecv.apply(packed, src_rank, link_group, device)


def pipeline_forward_distributed(my_stage, my_stage_idx, nhlevels, device,
                                 link_groups, pp_ranks, imod_bottom,
                                 data, state_labels, bcs, opts, parent):
    """Distributed pipeline forward with differentiable communication.

    Each rank runs its own stage.  Activations flow forward via
    pipeline_send_recv; gradients flow backward automatically through
    the custom autograd function.

    Args:
        my_stage: TurbTStage for this rank's level.
        my_stage_idx: int, which level this rank handles.
        nhlevels: int, total number of stages.
        device: this rank's device.
        link_groups: list of dist process groups, one per adjacent pair.
        pp_ranks: list of global ranks in this pipeline group.
        imod_bottom: int, coarsest active level.
        data: (T, B, C, D, H, W) input tensor.
        state_labels, bcs, opts, parent: as in TurbTIterative.forward.

    Returns:
        output tensor (B, C_out, D, H, W) on the tail rank, None otherwise.
    """
    # Only compute the filtered data needed for THIS rank's stage.
    # For stage imod, we need n_filt = nhlevels - 1 - imod filter applications.
    n_filt = parent.nhlevels - 1 - my_stage_idx
    my_filtered = data
    my_blockdict = copy.deepcopy(opts.blockdict)
    for _ in range(n_filt):
        my_filtered, my_blockdict = parent.filterdata(
            my_filtered, blockdict=copy.deepcopy(my_blockdict))

    my_stage.imod_bottom = imod_bottom
    my_stage.set_context(
        state_labels=state_labels, bcs=bcs,
        tkhead_name=opts.tkhead_name,
        blockdict=copy.deepcopy(my_blockdict),
        filtered_data=my_filtered,
        leadtime=opts.leadtime, cond_input=opts.cond_input,
        sequence_parallel_group=opts.sequence_parallel_group)
    del my_filtered  # free immediately after context is set

    # Head stage: create sentinel
    if my_stage_idx == imod_bottom:
        packed_in = pack_interstage(
            None,
            torch.zeros(1, device=device, dtype=data.dtype),
            torch.ones(1, device=device, dtype=data.dtype))
    else:
        # Receive from previous stage (differentiable)
        link_idx = my_stage_idx - 1
        prev_rank = pp_ranks[my_stage_idx - 1]
        # Placeholder — the actual data comes from the broadcast
        placeholder = torch.empty(0, device=device, dtype=data.dtype)
        packed_in = pipeline_send_recv(
            placeholder, src_rank=prev_rank,
            link_group=link_groups[link_idx], device=device)

    # Run my stage
    packed_out = my_stage(packed_in)

    # Send to next stage (differentiable)
    if my_stage_idx < nhlevels - 1:
        link_idx = my_stage_idx
        my_rank = pp_ranks[my_stage_idx]
        pipeline_send_recv(
            packed_out, src_rank=my_rank,
            link_group=link_groups[link_idx], device=device)

    if my_stage_idx == nhlevels - 1:
        x_pred, _, _, _ = unpack_interstage(packed_out)
        return x_pred
    else:
        return None

