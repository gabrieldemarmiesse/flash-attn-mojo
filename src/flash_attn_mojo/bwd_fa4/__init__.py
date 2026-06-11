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
            0, 0, 0, 0,  # varlen extras
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
            0, 0, 0, 0, 0,  # varlen extras
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
            0, 0, 0,  # varlen extras
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


def _build_bwd_tables(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_m: int,
    causal: bool,
) -> tuple:
    """Host work tables for the varlen bwd (vectorized torch on a
    host copy; one D2H sync).

    q-tile table int32[num_q_tiles, 8]: one row per main-kernel
    m-block — (m_local, q_row_base, seqlen_q, mpad_base, 0...).
    Shared by preprocess and convert.

    kv-tile table int32[num_kv_tiles, 8]: one row per main-kernel
    CTA — (n_block, q_row_base, k_row_base, seqlen_q, seqlen_k,
    num_m_blocks, m_start, mpad_base).

    Stats live in FA4's padded-packed layout: per-seq window i
    starts at padded_offset_i = ((cu_q[i] + i*BM)//BM)*BM (the i*BM
    slack guarantees non-overlap); total_qpad = num_mpad*BM with
    num_mpad = ceil((total_q + (nseq+1)*BM)/BM) per FA4's formula.
    """
    cu_q = cu_seqlens_q.detach().to("cpu", torch.int64)
    cu_k = cu_seqlens_k.detach().to("cpu", torch.int64)
    seqlens_q = cu_q[1:] - cu_q[:-1]
    seqlens_k = cu_k[1:] - cu_k[:-1]
    # Arbitrary lengths >= 1 (ragged kv tails are masked in-kernel
    # and tail tiles stored row-predicated); self-attn only.
    assert bool((seqlens_q > 0).all()), (
        "bwd_fa4_varlen needs every sequence length >= 1"
    )
    assert torch.equal(seqlens_q, seqlens_k), (
        "bwd_fa4_varlen is self-attn only (equal q/k lengths)"
    )
    nseq = len(seqlens_q)
    total_q = int(cu_q[-1])

    m_counts = (seqlens_q + block_m - 1) // block_m
    mpad_base = (cu_q[:-1] + torch.arange(nseq) * block_m) // block_m
    num_mpad = -(-(total_q + (nseq + 1) * block_m) // block_m)
    # Table fields are int32; .to(int32) wraps silently on overflow.
    assert num_mpad * block_m < 2**31 and int(cu_k[-1]) < 2**31
    # Window-fit (the slack formula guarantees this; assert anyway —
    # an overlap silently corrupts the next sequence's stats).
    ends = mpad_base + m_counts
    nxt = torch.cat(
        [mpad_base[1:], torch.tensor([num_mpad], dtype=torch.int64)]
    )
    assert bool((ends <= nxt).all()), "padded stat windows overlap"

    def expand(counts):
        n = int(counts.sum())
        sidx = torch.repeat_interleave(torch.arange(nseq), counts)
        starts = torch.cumsum(counts, 0) - counts
        local = torch.arange(n) - starts[sidx]
        return n, sidx, local

    num_q_tiles, q_sidx, q_local = expand(m_counts)
    qt = torch.zeros((num_q_tiles, 8), dtype=torch.int32)
    qt[:, 0] = q_local.to(torch.int32)
    qt[:, 1] = cu_q[:-1][q_sidx].to(torch.int32)
    qt[:, 2] = seqlens_q[q_sidx].to(torch.int32)
    qt[:, 3] = mpad_base[q_sidx].to(torch.int32)

    n_counts = (seqlens_k + _BLOCK - 1) // _BLOCK
    num_kv_tiles, k_sidx, n_local = expand(n_counts)
    kt = torch.zeros((num_kv_tiles, 8), dtype=torch.int32)
    kt[:, 0] = n_local.to(torch.int32)
    kt[:, 1] = cu_q[:-1][k_sidx].to(torch.int32)
    kt[:, 2] = cu_k[:-1][k_sidx].to(torch.int32)
    kt[:, 3] = seqlens_q[k_sidx].to(torch.int32)
    kt[:, 4] = seqlens_k[k_sidx].to(torch.int32)
    kt[:, 5] = m_counts[k_sidx].to(torch.int32)
    if causal:
        kt[:, 6] = ((n_local * _BLOCK) // block_m).to(torch.int32)
    kt[:, 7] = mpad_base[k_sidx].to(torch.int32)

    dev = cu_seqlens_q.device
    return (
        qt.to(dev), num_q_tiles, kt.to(dev), num_kv_tiles,
        int(num_mpad), total_q, int(cu_k[-1]),
    )


def bwd_fa4_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed varlen backward (dq, dk, dv).

    q/k/v/out/dout: (total_tokens, H, D) bf16 contiguous, D=128.
    lse: (H, total_q) fp32 (the packed natural-log LSE from the
    varlen fwd). MHA only; arbitrary sequence lengths >= 1;
    self-attn lengths.
    """
    from flash_attn_mojo.bwd_fa4._jit import (
        call_bwd_fa4_convert,
        call_bwd_fa4_main,
        call_bwd_fa4_preprocess,
        make_config,
    )

    total_q, nheads, head_dim = q.shape
    total_k = k.shape[0]
    nheads_kv = k.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    assert q.dtype == torch.bfloat16, "bwd_fa4 is bf16-only"
    assert head_dim == 128, "bwd_fa4 is head_dim=128-only"
    assert nheads_kv == nheads, "bwd_fa4_varlen v1 is MHA-only"
    assert k.shape == (total_k, nheads_kv, head_dim)
    assert v.shape == k.shape
    assert lse.shape == (nheads, total_q) and lse.dtype == torch.float32
    for t in (q, k, v, out, dout, lse):
        assert t.is_contiguous()
    for cu in (cu_seqlens_q, cu_seqlens_k):
        assert cu.dtype == torch.int32 and cu.is_cuda and cu.is_contiguous()

    block_m = 64 if causal else _BLOCK_M
    (
        q_table, num_q_tiles, kv_table, num_kv_tiles,
        num_mpad, total_q_, total_k_,
    ) = _build_bwd_tables(cu_seqlens_q, cu_seqlens_k, block_m, causal)
    assert total_q_ == total_q and total_k_ == total_k
    total_qpad = num_mpad * block_m

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    # Aux work-table row (index = grid_dim.x): the raw dk/dv base
    # addresses as two int64s, for the kernel's row-predicated
    # ragged-tail stores.
    aux = (
        torch.tensor(
            [dk.data_ptr(), dv.data_ptr(), 0, 0], dtype=torch.int64
        )
        .view(torch.int32)
        .reshape(1, 8)
        .to(q.device)
    )
    kv_table = torch.cat([kv_table, aux], dim=0).contiguous()
    dpsum = torch.empty(
        (nheads, total_qpad), dtype=torch.float32, device=q.device
    )
    lse_log2 = torch.empty_like(dpsum)
    dq_accum = torch.empty(
        (nheads * total_qpad * head_dim,),
        dtype=torch.float32,
        device=q.device,
    )

    config = make_config(
        _DTYPE_CODE[q.dtype], head_dim, True, causal, 1, True
    )
    stream = torch.cuda.current_stream().cuda_stream
    nseq = cu_seqlens_q.numel() - 1

    call_bwd_fa4_preprocess(
        (
            nseq,
            total_q,  # seqlen slot (unused by the varlen kernel path)
            nheads,
            out.data_ptr(),
            dout.data_ptr(),
            lse.data_ptr(),
            dpsum.data_ptr(),
            lse_log2.data_ptr(),
            dq_accum.data_ptr(),
            0,
            0,
            stream,
            num_q_tiles,
            q_table.data_ptr(),
            total_q,
            total_qpad,
        ),
        config,
    )
    call_bwd_fa4_main(
        (
            nseq,
            total_q,
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
            num_kv_tiles,
            kv_table.data_ptr(),
            total_q,
            total_k,
            num_mpad,
        ),
        config,
    )
    call_bwd_fa4_convert(
        (
            nseq,
            total_q,
            nheads,
            float(softmax_scale),
            dq_accum.data_ptr(),
            dq.data_ptr(),
            stream,
            num_q_tiles,
            q_table.data_ptr(),
            num_mpad,
        ),
        config,
    )
    return dq, dk, dv
