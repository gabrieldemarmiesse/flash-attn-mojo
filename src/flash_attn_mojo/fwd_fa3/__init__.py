"""FA3 (sm_90+) forward subpackage: TMA + WGMMA kernel.

Parallel to ``flash_attn_mojo.fwd`` (FA2). The Python wrapper picks
which one to call at runtime based on the GPU compute capability and
input shape — see ``flash_attn_mojo._fn``.

MVP scope: bf16, head_dim=64, non-causal, no dropout, no MQA, no
softcap, no ALiBi, no window, no LSE return. Other configs raise
NotImplementedError so the caller can fall back.
"""

from __future__ import annotations

import torch

from flash_attn_mojo._dtype import _DTYPE_CODE


def native_fwd_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None:
    """JIT-compile (if needed) and dispatch a single FA3 fwd call.

    MVP signature — only the tensors and the softmax scale. Causal,
    softcap, ALiBi, dropout, LSE return, MQA/GQA, window, varlen are
    all out of scope for this MVP and the dispatcher in
    ``flash_attn_mojo._fn`` should route those calls to the FA2 path.

    q, k, v, out: (batch, seqlen, nheads, head_dim) bf16 contiguous
        tensors with head_dim=64. nheads_q must equal nheads_k for now.
    softmax_scale: scalar applied to Q·Kᵀ before softmax.
    """
    from flash_attn_mojo.fwd_fa3._jit import call_fwd_fa3

    batch, seqlen, nheads, head_dim = q.shape

    call_fwd_fa3(
        (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            out.data_ptr(),
            batch,
            seqlen,
            nheads,
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
            out.stride(0),
            out.stride(1),
            out.stride(2),
            torch.cuda.current_stream().cuda_stream,
            _DTYPE_CODE[q.dtype],
            head_dim,
            1,  # use_external_stream
        )
    )
