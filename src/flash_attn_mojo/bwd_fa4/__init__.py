"""FA4-target backward subpackage: Hopper TMA + WGMMA kernels.

Chasing Tri Dao's FlashAttention-4 sm90 backward (see
`reference_ptx/README.md`). Scope: bf16, head_dim=128, non-causal,
fixed seqlen (S % 128 == 0), contiguous (B, S, H, D), Hq == Hk.
"""

from __future__ import annotations

import math

import torch

from flash_attn_mojo._dtype import _DTYPE_CODE

_BLOCK = 128
# Main-kernel Q-tile rows (FA4's tile_m for hdim128 non-causal).
# Side buffers are padded to a multiple of this; the preprocess
# kernel fills the pad with lse=+inf / dpsum=0 so the main kernel's
# tail m-block contributes exactly zero.
_BLOCK_M = 80


def bwd_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (dq, dk, dv). Mirrors
    ``flash_attn.cute.interface._flash_attn_bwd``'s simplest call.

    q/k/v/out/dout: (B, S, H, D) bf16 contiguous, D=128, S % 128 == 0.
    lse: (B, H, S) fp32 contiguous (natural-log LSE from the fwd).
    """
    from flash_attn_mojo.bwd_fa4._jit import (
        call_bwd_fa4_convert,
        call_bwd_fa4_main,
        call_bwd_fa4_preprocess,
        make_config,
    )

    batch, seqlen, nheads, head_dim = q.shape
    nheads_kv = k.shape[2]
    gqa_ratio = nheads // nheads_kv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    assert q.dtype == torch.bfloat16, "bwd_fa4 is bf16-only"
    assert head_dim == 128, "bwd_fa4 is head_dim=128-only"
    assert seqlen % _BLOCK == 0, "bwd_fa4 needs seqlen % 128 == 0"
    assert k.shape == (batch, seqlen, nheads_kv, head_dim)
    assert v.shape == k.shape
    assert nheads % nheads_kv == 0, "Hq must be a multiple of Hkv"
    assert lse.shape == (batch, nheads, seqlen) and lse.dtype == torch.float32
    for t in (q, k, v, out, dout):
        assert t.is_contiguous()
    assert lse.is_contiguous()

    dq = torch.empty_like(q)
    if gqa_ratio > 1:
        # fp32 accumulators (B, Hkv, S, D): every q-head CTA of a
        # group bulk-reduce-adds its dK/dV into them; converted to
        # bf16 (B, S, Hkv, D) below.
        dk_accum = torch.empty(
            (batch, nheads_kv, seqlen, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        dv_accum = torch.empty_like(dk_accum)
        dk_main_addr = dk_accum.data_ptr()
        dv_main_addr = dv_accum.data_ptr()
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
    else:
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dk_main_addr = dk.data_ptr()
        dv_main_addr = dv.data_ptr()
    # Causal uses FA4's tile_m=64 (divides any supported seqlen: no
    # padding); non-causal tile_m=80 pads.
    block_m = 64 if causal else _BLOCK_M
    seqlen_pad = -(-seqlen // block_m) * block_m
    dpsum = torch.empty(
        (batch, nheads, seqlen_pad), dtype=torch.float32, device=q.device
    )
    lse_log2 = torch.empty_like(dpsum)
    # Opaque blocked fragment dump (FA4's trick): per (b, h, m-block
    # of 80 rows) a contiguous [wg(2)][chunk(10)][tid(128)][4] f32
    # region, bulk-reduce-added by the main kernel's drain warp and
    # decoded by the convert kernel. Numel = B * H * Spad * D.
    dq_accum = torch.empty(
        (batch * nheads * seqlen_pad * head_dim,),
        dtype=torch.float32,
        device=q.device,
    )

    config = make_config(
        _DTYPE_CODE[q.dtype], head_dim, True, causal, gqa_ratio
    )
    stream = torch.cuda.current_stream().cuda_stream

    call_bwd_fa4_preprocess(
        (
            batch,
            seqlen,
            nheads,
            out.data_ptr(),
            dout.data_ptr(),
            lse.data_ptr(),
            dpsum.data_ptr(),
            lse_log2.data_ptr(),
            dq_accum.data_ptr(),
            dk_main_addr if gqa_ratio > 1 else 0,
            dv_main_addr if gqa_ratio > 1 else 0,
            stream,
        ),
        config,
    )
    call_bwd_fa4_main(
        (
            batch,
            seqlen,
            nheads,
            float(softmax_scale),
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            dout.data_ptr(),
            dk_main_addr,
            dv_main_addr,
            lse_log2.data_ptr(),
            dpsum.data_ptr(),
            dq_accum.data_ptr(),
            stream,
        ),
        config,
    )
    call_bwd_fa4_convert(
        (
            batch,
            seqlen,
            nheads,
            float(softmax_scale),
            dq_accum.data_ptr(),
            dq.data_ptr(),
            stream,
        ),
        config,
    )
    if gqa_ratio > 1:
        # (B, Hkv, S, D) fp32 -> (B, S, Hkv, D) bf16, one fused
        # copy-cast each (the timed FA4 path runs its own
        # postprocess for the same conversion).
        dk.copy_(dk_accum.permute(0, 2, 1, 3))
        dv.copy_(dv_accum.permute(0, 2, 1, 3))
    return dq, dk, dv
