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
    window_left: int = 0,
    softcap: float = 0.0,
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
    native_fwd_fa4(
        q, k, v, out, lse, softmax_scale, causal, window_left, softcap
    )
    return out, lse


def native_fwd_fa4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
    window_left: int = 0,
    softcap: float = 0.0,
) -> None:
    """JIT-compile (if needed) and dispatch a single fwd call."""
    from flash_attn_mojo.fwd_fa4._jit import call_fwd_fa4

    batch, seqlen, nheads, head_dim = q.shape
    nheads_kv = k.shape[2]
    assert q.dtype in (torch.bfloat16, torch.float16), (
        "fwd_fa4 is bf16/fp16-only"
    )
    assert head_dim in (64, 128), "fwd_fa4 supports head_dim 64/128"
    assert seqlen % _BLOCK_N == 0, "fwd_fa4 needs seqlen % 128 == 0"
    if window_left:
        assert causal and head_dim == 128, (
            "window: causal + head_dim=128 (any left >= 1)"
        )
    # The cap is comptime (one JIT variant per value); x1000 keeps
    # the define an int while representing Gemma-class caps exactly.
    softcap_x1000 = round(float(softcap) * 1000)
    if softcap_x1000:
        assert softcap > 0 and head_dim == 128, (
            "softcap v1: positive cap + head_dim=128"
        )
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
            1 if window_left else 0,  # window (comptime)
            window_left,
            softcap_x1000,  # comptime
        )
    )


_BLOCK_M = 128  # fwd m-tile (kFa4BlockM)


def _effective_lens(
    cu: torch.Tensor, seqused: torch.Tensor | None, name: str
) -> torch.Tensor:
    """Per-sequence lengths: cu_seqlens deltas, overridden by
    seqused when given (the cu bases still define the memory
    layout — seqused just shortens each sequence's used prefix)."""
    lens = cu[1:] - cu[:-1]
    if seqused is None:
        return lens
    su = seqused.detach().to("cpu", torch.int64)
    assert su.shape == lens.shape, f"{name} must be (nseq,)"
    assert bool((su >= 1).all() and (su <= lens).all()), (
        f"{name} must satisfy 1 <= {name}[i] <= seqlen[i]"
    )
    return su


def _build_fwd_tile_table(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    window_left: int = 0,
) -> tuple[torch.Tensor, int, int]:
    """Host work-item table: one int32[8] row per CTA m-tile —
    (m_block, q_row_base, k_row_base, seqlen_q, seqlen_k, 0, 0, 0).
    Built with vectorized torch on a host copy of cu_seqlens (one
    D2H sync). Returns (device table, num_tiles, max_seqlen_q)."""
    cu_q = cu_seqlens_q.detach().to("cpu", torch.int64)
    cu_k = cu_seqlens_k.detach().to("cpu", torch.int64)
    # Table fields are int32 row indices; .to(int32) wraps silently.
    assert int(cu_q[-1]) < 2**31 and int(cu_k[-1]) < 2**31
    seqlens_q = _effective_lens(cu_q, seqused_q, "seqused_q")
    seqlens_k = _effective_lens(cu_k, seqused_k, "seqused_k")
    # Arbitrary lengths >= 1 (ragged tails are masked in-kernel and
    # tail tiles stored row-predicated); empty sequences are out of
    # the envelope.
    assert bool((seqlens_q > 0).all() and (seqlens_k > 0).all()), (
        "fa4_varlen_fwd needs every sequence length >= 1"
    )
    if causal:
        # Bottom-right diagonal: slq > slk would leave the first
        # (slq - slk) q rows attending nothing (out=0/lse=-inf
        # semantics) — not in the v1 envelope.
        assert bool((seqlens_q <= seqlens_k).all()), (
            "causal varlen cross-attention requires seqlen_q <= "
            "seqlen_k per sequence"
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
    if window_left:
        # sched_swizzle carries this table's address under varlen,
        # so win_left rides the free col 5 instead of the LPT slot.
        table[:, 5] = window_left
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
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    softcap: float = 0.0,
    window_left: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed varlen forward. Mirrors
    ``flash_attn.cute.flash_attn_varlen_func``'s simplest call.

    q/k/v: (total_tokens, H, D) bf16 contiguous, D=128.
    cu_seqlens_{q,k}: (nseq+1,) int32 CUDA, monotone, [0] == 0.
    Returns (out, lse) with lse (H, total_q) f32 (packed, FA4's
    varlen layout). Arbitrary sequence lengths >= 1, self- or
    cross-attention (causal cross uses FA4's bottom-right diagonal
    and requires seqlen_q <= seqlen_k per sequence).
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
    if window_left:
        assert causal and window_left >= 1, (
            "varlen window: causal + left >= 1"
        )

    table, num_tiles, max_seqlen_q = _build_fwd_tile_table(
        cu_seqlens_q, cu_seqlens_k, causal, seqused_q, seqused_k,
        window_left,
    )

    # With seqused the rows past each sequence's used prefix are
    # never written — zero-fill so out is fully defined (FA4 leaves
    # them undefined; a memset is cheap and safer).
    out = (
        torch.zeros_like(q)
        if seqused_q is not None
        else torch.empty_like(q)
    )
    lse_alloc = torch.zeros if seqused_q is not None else torch.empty
    lse = lse_alloc(
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
            1 if window_left else 0,  # window (comptime)
            window_left,  # (keys window_unaligned; rides table col 5)
            round(float(softcap) * 1000),  # softcap_x1000 (comptime)
        )
    )
    return out, lse
