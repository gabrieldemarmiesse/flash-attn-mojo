"""GPU backward subpackage: kernel + JIT dispatcher + Python wrapper.

STATUS: MVP main bwd kernel (`native_bwd_main`) added alongside the
existing `native_bwd_preprocess`. The MVP envelope is bf16 / head_dim=64
/ non-causal / no MQA / no softcap-alibi-window-dropout / equal seqlen.
Outside that envelope (and by default until correctness is verified)
the autograd backward in `_fn.py` still falls back to the pure-pytorch
reference path.
"""

from __future__ import annotations

import torch

from flash_attn_mojo._dtype import _DTYPE_CODE


def native_bwd_preprocess(
    dout: torch.Tensor,
    out: torch.Tensor,
    delta: torch.Tensor,
    dqaccum: torch.Tensor,
) -> None:
    """JIT-compile (if needed) and dispatch the bwd-preprocess kernel.

    Does two things, mirroring Tri Dao's flash_bwd_preprocess pipeline:
      1. ``delta[b, h, q] = sum_d dO[b, q, h, d] * O[b, q, h, d]`` in
         fp32 (the accumulator is always fp32 regardless of input
         dtype).
      2. Zeroes ``dqaccum[b, h, q, d]`` — the fp32 workspace the main
         bwd kernel atomically accumulates dQ into across KV blocks.

    dout, out: (batch, seqlen, nheads, head_dim) tensors. Same dtype
        and shape as the fwd's ``out`` tensor.
    delta: (batch, nheads, seqlen) fp32 tensor — must be contiguous so
        the kernel can index it as ``delta[b * H * L + h * L + q]``.
    dqaccum: (batch, nheads, seqlen, head_dim) fp32 workspace. The
        kernel writes zeros to every element; the caller does not need
        to pre-initialise it.
    """
    from flash_attn_mojo.bwd._preprocess_jit import call_bwd_preprocess

    assert dout.dtype == out.dtype, "dout and out must share dtype"
    assert delta.dtype == torch.float32, "delta must be fp32"
    assert delta.is_contiguous(), "delta must be contiguous"
    assert dqaccum.dtype == torch.float32, "dqaccum must be fp32"

    batch, seqlen, nheads, head_dim = dout.shape
    assert tuple(out.shape) == (batch, seqlen, nheads, head_dim), (
        "out shape must match dout"
    )
    assert tuple(delta.shape) == (batch, nheads, seqlen), (
        "delta shape must be (batch, nheads, seqlen)"
    )
    assert tuple(dqaccum.shape) == (batch, nheads, seqlen, head_dim), (
        "dqaccum shape must be (batch, nheads, seqlen, head_dim)"
    )

    call_bwd_preprocess(
        (
            dout.data_ptr(),
            out.data_ptr(),
            delta.data_ptr(),
            dqaccum.data_ptr(),
            batch,
            seqlen,
            nheads,
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            delta.stride(0),
            delta.stride(1),
            dqaccum.stride(0),
            dqaccum.stride(1),
            dqaccum.stride(2),
            torch.cuda.current_stream().cuda_stream,
            _DTYPE_CODE[dout.dtype],
            head_dim,
            1,  # use_external_stream
        )
    )


def native_bwd_main(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dqaccum: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
) -> None:
    """JIT-compile (if needed) and dispatch the main bwd kernel.

    MVP envelope: bf16, head_dim=64. MQA/GQA supported (nheads_q must
    be a multiple of nheads_kv). Outputs dk/dv are written directly;
    dq is accumulated into ``dqaccum`` (fp32) and must be converted
    to dq's dtype by the caller (the convert_dq kernel).

    q, dout:       (B, L, Hq, D) bf16.
    k, v:          (B, L, Hkv, D) bf16. Hq % Hkv == 0.
    lse, delta:    (B, Hq, L) fp32.
    dk, dv:        (B, L, Hkv, D) bf16 — same shape as k, v.
    dqaccum:       (B, Hq, L, D) fp32 — pre-zeroed by preprocess kernel.
    """
    from flash_attn_mojo.bwd._main_jit import call_bwd_main

    assert q.dtype == k.dtype == v.dtype == dout.dtype, (
        "q, k, v, dout must share dtype"
    )
    assert lse.dtype == torch.float32 and delta.dtype == torch.float32
    assert dqaccum.dtype == torch.float32
    assert lse.is_contiguous() and delta.is_contiguous()

    batch, seqlen, nheads_q, head_dim = q.shape
    nheads_kv = k.shape[2]
    assert nheads_q % nheads_kv == 0, (
        f"nheads_q ({nheads_q}) must be a multiple of nheads_kv ({nheads_kv})"
    )

    call_bwd_main(
        (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            dout.data_ptr(),
            lse.data_ptr(),
            delta.data_ptr(),
            dk.data_ptr(),
            dv.data_ptr(),
            dqaccum.data_ptr(),
            batch,
            seqlen,
            nheads_q,
            nheads_kv,
            float(softmax_scale),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            lse.stride(0),
            lse.stride(1),
            delta.stride(0),
            delta.stride(1),
            dqaccum.stride(0),
            dqaccum.stride(1),
            dqaccum.stride(2),
            torch.cuda.current_stream().cuda_stream,
            _DTYPE_CODE[q.dtype],
            head_dim,
            1 if causal else 0,
            1,  # use_external_stream
        )
    )


def native_bwd_convert_dq(
    dqaccum: torch.Tensor,
    dq: torch.Tensor,
) -> None:
    """JIT-compile (if needed) and dispatch the bwd convert-dQ kernel.

    Casts ``dqaccum`` (fp32, (B, H, L, D)) to ``dq``'s dtype (bf16 or
    fp16) and writes into ``dq`` ((B, L, H, D)) — mirrors Tri Dao's
    flash_bwd_convert_dq_kernel. Rows past ``seq_len`` in any q-tile
    tail are skipped.
    """
    from flash_attn_mojo.bwd._convert_dq_jit import call_bwd_convert_dq

    assert dqaccum.dtype == torch.float32, "dqaccum must be fp32"
    assert dq.dtype in (torch.bfloat16, torch.float16), (
        f"dq dtype must be bf16 or fp16, got {dq.dtype}"
    )

    batch, seqlen, nheads, head_dim = dq.shape
    assert tuple(dqaccum.shape) == (batch, nheads, seqlen, head_dim), (
        "dqaccum shape must be (batch, nheads, seqlen, head_dim)"
    )

    call_bwd_convert_dq(
        (
            dqaccum.data_ptr(),
            dq.data_ptr(),
            batch,
            seqlen,
            nheads,
            dqaccum.stride(0),
            dqaccum.stride(1),
            dqaccum.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            torch.cuda.current_stream().cuda_stream,
            _DTYPE_CODE[dq.dtype],
            head_dim,
            1,  # use_external_stream
        )
    )
