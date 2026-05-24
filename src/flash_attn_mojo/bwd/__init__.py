"""GPU backward subpackage: kernel + JIT dispatcher + Python wrapper.

STATUS: partial — `native_bwd_preprocess` is implemented (computes
`delta = rowsum(dO * O)` on-device). The full `native_bwd` (dq, dk, dv
matmuls) is still a TODO; the autograd backward in `_fn.py` currently
calls into the preprocess kernel for `delta` and falls back to pytorch
for the rest.
"""

from __future__ import annotations

import torch

from flash_attn_mojo._dtype import _DTYPE_CODE


def native_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
) -> None:
    """JIT-compile + dispatch the GPU backward kernel.

    Placeholder until the full bwd kernel is implemented.
    """
    raise NotImplementedError(
        "flash_attn_mojo.bwd.native_bwd: kernel not implemented yet"
    )


def native_bwd_preprocess(
    dout: torch.Tensor,
    out: torch.Tensor,
    delta: torch.Tensor,
) -> None:
    """JIT-compile (if needed) and dispatch the bwd-preprocess kernel.

    Computes ``delta[b, h, q] = sum_d dO[b, q, h, d] * O[b, q, h, d]``
    in fp32 (the accumulator is always fp32 regardless of input dtype).

    dout, out: (batch, seqlen, nheads, head_dim) tensors. Same dtype
        and shape as the fwd's ``out`` tensor.
    delta: (batch, nheads, seqlen) fp32 tensor — must be contiguous so
        the kernel can index it as ``delta[b * H * L + h * L + q]``.
    """
    from flash_attn_mojo.bwd._preprocess_jit import call_bwd_preprocess

    assert dout.dtype == out.dtype, "dout and out must share dtype"
    assert delta.dtype == torch.float32, "delta must be fp32"
    assert delta.is_contiguous(), "delta must be contiguous"

    batch, seqlen, nheads, head_dim = dout.shape
    assert tuple(out.shape) == (batch, seqlen, nheads, head_dim), (
        "out shape must match dout"
    )
    assert tuple(delta.shape) == (batch, nheads, seqlen), (
        "delta shape must be (batch, nheads, seqlen)"
    )

    call_bwd_preprocess(
        (
            dout.data_ptr(),
            out.data_ptr(),
            delta.data_ptr(),
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
            torch.cuda.current_stream().cuda_stream,
            _DTYPE_CODE[dout.dtype],
            head_dim,
            1,  # use_external_stream
        )
    )
