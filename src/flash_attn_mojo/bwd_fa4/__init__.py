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


def bwd_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float | None = None,
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
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    assert q.dtype == torch.bfloat16, "bwd_fa4 v1 is bf16-only"
    assert head_dim == 128, "bwd_fa4 v1 is head_dim=128-only"
    assert seqlen % _BLOCK == 0, "bwd_fa4 v1 needs seqlen % 128 == 0"
    assert k.shape == q.shape and v.shape == q.shape, "Hq must equal Hk"
    assert lse.shape == (batch, nheads, seqlen) and lse.dtype == torch.float32
    for t in (q, k, v, out, dout):
        assert t.is_contiguous()
    assert lse.is_contiguous()

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dpsum = torch.empty(
        (batch, nheads, seqlen), dtype=torch.float32, device=q.device
    )
    lse_log2 = torch.empty_like(dpsum)
    # Opaque blocked fragment dump (FA4's trick): per (b, h, m-block
    # of 64 rows) a contiguous [wg(2)][chunk(8)][tid(128)][4] f32
    # region, bulk-reduce-added by the main kernel's drain warp and
    # decoded by the convert kernel. Same numel as (B, S, H, D).
    dq_accum = torch.empty(
        (batch * nheads * seqlen * head_dim,),
        dtype=torch.float32,
        device=q.device,
    )

    config = make_config(_DTYPE_CODE[q.dtype], head_dim, True)
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
            dk.data_ptr(),
            dv.data_ptr(),
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
    return dq, dk, dv
