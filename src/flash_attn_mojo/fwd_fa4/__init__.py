"""FA4-target forward subpackage: Hopper TMA + WGMMA kernel.

The kernel this package is chasing is Tri Dao's FlashAttention-4
sm90 forward (see ``reference_ptx/README.md``). Scope is the FA4
"simplest call": bf16, head_dim=128, non-causal, fixed seqlen,
contiguous (B, L, H, D), Hq == Hk. Everything else is out of scope
until perf parity is reached.
"""

from __future__ import annotations

import math

import torch

from flash_attn_mojo._dtype import _DTYPE_CODE

_BLOCK_N = 128


def fa4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: allocate out + lse, dispatch, return both.

    Mirrors ``flash_attn.cute.flash_attn_func(q, k, v)``: returns
    (out, lse) with lse (B, H, L) fp32 natural-log row LSE (what the
    backward consumes). q, k, v: (B, L, H, D) bf16 contiguous,
    D=128, L % 128 == 0.
    """
    batch, seqlen, nheads, head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    lse = torch.empty(
        (batch, nheads, seqlen), dtype=torch.float32, device=q.device
    )
    native_fwd_fa4(q, k, v, out, lse, softmax_scale, causal)
    return out, lse


def native_fwd_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
) -> None:
    """JIT-compile (if needed) and dispatch a single fwd call."""
    from flash_attn_mojo.fwd_fa4._jit import call_fwd_fa4

    batch, seqlen, nheads, head_dim = q.shape
    nheads_kv = k.shape[2]
    assert q.dtype == torch.bfloat16, "fwd_fa4 is bf16-only"
    assert head_dim == 128, "fwd_fa4 is head_dim=128-only"
    assert seqlen % _BLOCK_N == 0, "fwd_fa4 needs seqlen % 128 == 0"
    assert k.shape == (batch, seqlen, nheads_kv, head_dim)
    assert v.shape == k.shape
    assert nheads % nheads_kv == 0, "Hq must be a multiple of Hkv"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert out.is_contiguous()
    assert lse.shape == (batch, nheads, seqlen) and lse.dtype == torch.float32
    assert lse.is_contiguous()

    call_fwd_fa4(
        (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
            lse.data_ptr(),
            batch,
            seqlen,
            nheads,
            float(softmax_scale),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            torch.cuda.current_stream().cuda_stream,
            _DTYPE_CODE[q.dtype],
            head_dim,
            1,  # use_external_stream
            1 if causal else 0,
            nheads // nheads_kv,  # gqa_ratio
            0,  # varlen
        )
    )
