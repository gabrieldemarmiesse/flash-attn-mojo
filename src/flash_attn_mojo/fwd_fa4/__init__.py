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
    assert q.dtype in (torch.bfloat16, torch.float16), (
        "fwd_fa4 is bf16/fp16-only"
    )
    assert head_dim in (64, 128), "fwd_fa4 supports head_dim 64/128"
    assert not (head_dim == 64 and q.dtype == torch.float16), (
        "hdim64 fp16 needs the n=64 RS wgmma arm"
    )
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
            0,  # varlen total_q
            0,  # varlen total_k
            0,  # varlen tile-table addr
            0,  # varlen num_tiles
        )
    )


_BLOCK_M = 128  # fwd m-tile (kFa4BlockM)


def _build_fwd_tile_table(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    """Host work-item table: one int32[8] row per CTA m-tile —
    (m_block, q_row_base, k_row_base, seqlen_q, seqlen_k, 0, 0, 0).
    Built with vectorized torch on a host copy of cu_seqlens (one
    D2H sync). Returns (device table, num_tiles, max_seqlen_q)."""
    cu_q = cu_seqlens_q.detach().to("cpu", torch.int64)
    cu_k = cu_seqlens_k.detach().to("cpu", torch.int64)
    # Table fields are int32 row indices; .to(int32) wraps silently.
    assert int(cu_q[-1]) < 2**31 and int(cu_k[-1]) < 2**31
    seqlens_q = cu_q[1:] - cu_q[:-1]
    seqlens_k = cu_k[1:] - cu_k[:-1]
    # Arbitrary lengths >= 1 (ragged tails are masked in-kernel and
    # tail tiles stored row-predicated); empty sequences are out of
    # the envelope.
    assert bool((seqlens_q > 0).all() and (seqlens_k > 0).all()), (
        "fa4_varlen_fwd needs every sequence length >= 1"
    )
    m_counts = (seqlens_q + _BLOCK_M - 1) // _BLOCK_M
    num_tiles = int(m_counts.sum())
    sidx = torch.repeat_interleave(torch.arange(len(seqlens_q)), m_counts)
    starts = torch.cumsum(m_counts, 0) - m_counts
    table = torch.zeros((num_tiles, 8), dtype=torch.int32)
    table[:, 0] = (torch.arange(num_tiles) - starts[sidx]).to(torch.int32)
    table[:, 1] = cu_q[:-1][sidx].to(torch.int32)
    table[:, 2] = cu_k[:-1][sidx].to(torch.int32)
    table[:, 3] = seqlens_q[sidx].to(torch.int32)
    table[:, 4] = seqlens_k[sidx].to(torch.int32)
    return (
        table.to(cu_seqlens_q.device),
        num_tiles,
        int(seqlens_q.max()),
    )


def fa4_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed varlen forward. Mirrors
    ``flash_attn.cute.flash_attn_varlen_func``'s simplest call.

    q/k/v: (total_tokens, H, D) bf16 contiguous, D=128.
    cu_seqlens_{q,k}: (nseq+1,) int32 CUDA, monotone, [0] == 0.
    Returns (out, lse) with lse (H, total_q) f32 (packed, FA4's
    varlen layout). Arbitrary sequence lengths >= 1; self-attn
    lengths (cu_seqlens_q == cu_seqlens_k elementwise).
    """
    from flash_attn_mojo.fwd_fa4._jit import call_fwd_fa4

    total_q, nheads, head_dim = q.shape
    total_k, nheads_kv, _ = k.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    assert q.dtype in (torch.bfloat16, torch.float16), (
        "fwd_fa4 is bf16/fp16-only"
    )
    assert head_dim == 128, "fwd_fa4 is head_dim=128-only"
    assert k.shape == (total_k, nheads_kv, head_dim)
    assert v.shape == k.shape
    assert nheads % nheads_kv == 0, "Hq must be a multiple of Hkv"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    for cu in (cu_seqlens_q, cu_seqlens_k):
        assert cu.dtype == torch.int32 and cu.is_cuda and cu.is_contiguous()
    assert cu_seqlens_q.shape == cu_seqlens_k.shape

    table, num_tiles, max_seqlen_q = _build_fwd_tile_table(
        cu_seqlens_q, cu_seqlens_k
    )

    out = torch.empty_like(q)
    lse = torch.empty(
        (nheads, total_q), dtype=torch.float32, device=q.device
    )
    nseq = cu_seqlens_q.numel() - 1
    call_fwd_fa4(
        (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
            lse.data_ptr(),
            nseq,
            max_seqlen_q,
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
            1,  # varlen
            total_q,
            total_k,
            table.data_ptr(),
            num_tiles,
        )
    )
    return out, lse
