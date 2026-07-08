"""Apple-GPU (Metal) forward subpackage: 8x8 simdgroup-matrix kernel.

The kernel (``kernel.mojo``) is at ccv/metal-flash-attention kernel
parity on M1–M4 (see METAL_PLAN.md). This wrapper bridges torch to it.

Bridging note: mojo's Metal runtime cannot bind torch's MPS buffers
(separate allocators; no shared registry — verified), so unlike the
CUDA path this does NOT run zero-copy over torch's device pointers.
Inputs are staged on CPU (contiguous fp16, head-major) and the mojo
launcher copies them through its own Metal buffers. The KERNEL is at
ccv/MFA parity (see METAL_PLAN.md), but this wrapper's end-to-end
wall time currently carries substantial staging overhead (~40–65% on
top of the kernel at S=1k–8k): per-call Metal buffer alloc/free plus
the torch↔mojo copy round-trips. Reaching the bench's kernel-only
speed from PyTorch needs a zero-copy buffer bridge (or at least
pooled, reused Metal buffers) — tracked as a follow-up. Correctness
and the fast kernel are done; the wrapper is functional, not yet
copy-optimal.

Envelope: fp16/bf16 q/k/v (computed in fp16), head_dim in {64, 128},
seqlen % 128 == 0, MHA or GQA (Hq % Hkv == 0; GQA is expanded to MHA
host-side). fp32 O out; fp32 natural-log LSE.
"""

from __future__ import annotations

import math

import torch

_SEQLEN_MULTIPLE = 128
_SUPPORTED_HEAD_DIMS = (64, 128)
_DTYPE_CODE_FP16 = 0


def fa_metal_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward attention on the Apple GPU. Returns (out, lse).

    q: (B, S, Hq, D); k, v: (B, S, Hkv, D). out: (B, S, Hq, D) in q's
    dtype; lse: (B, Hq, S) fp32 natural-log row logsumexp (matches the
    reference / the CUDA path's layout).
    """
    batch, seqlen, nheads, head_dim = q.shape
    nheads_kv = k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    assert q.dtype in (torch.float16, torch.bfloat16), (
        "fwd_metal is fp16/bf16 (computed in fp16)"
    )
    assert head_dim in _SUPPORTED_HEAD_DIMS, (
        f"fwd_metal supports head_dim in {_SUPPORTED_HEAD_DIMS}"
    )
    assert seqlen % _SEQLEN_MULTIPLE == 0, (
        "fwd_metal needs seqlen % 128 == 0"
    )
    assert k.shape == (batch, seqlen, nheads_kv, head_dim)
    assert v.shape == k.shape
    assert nheads % nheads_kv == 0, "Hq must be a multiple of Hkv"

    # Stage on CPU, fp16, head-major (B, H, S, D) contiguous — the
    # layout the kernel indexes (head = b*Hq + h, per-head (S, D) slab).
    # GQA is expanded to MHA here (cheap; the kernel stays MHA-only).
    def _stage(t: torch.Tensor, heads: int) -> torch.Tensor:
        if heads != nheads:
            t = t.repeat_interleave(nheads // heads, dim=2)
        return (
            t.detach()
            .to(device="cpu", dtype=torch.float16)
            .transpose(1, 2)
            .contiguous()
        )

    q_h = _stage(q, nheads)
    k_h = _stage(k, nheads_kv)
    v_h = _stage(v, nheads_kv)
    o_h = torch.empty(
        (batch, nheads, seqlen, head_dim), dtype=torch.float32
    )
    lse_h = torch.empty(
        (batch, nheads, seqlen), dtype=torch.float32
    )

    from flash_attn_mojo.fwd_metal._jit import call_fwd_metal

    call_fwd_metal(
        (
            q_h.data_ptr(),
            k_h.data_ptr(),
            v_h.data_ptr(),
            o_h.data_ptr(),
            lse_h.data_ptr(),
            batch,
            seqlen,
            nheads,
            float(softmax_scale),
            _DTYPE_CODE_FP16,
            head_dim,
        )
    )
    # Keep the staged host tensors alive until the kernel + copybacks
    # finish (call_fwd_metal synchronizes internally).
    _ = (q_h, k_h, v_h)

    out = (
        o_h.transpose(1, 2)  # (B, S, H, D)
        .contiguous()
        .to(device=q.device, dtype=q.dtype)
    )
    lse = lse_h.to(device=q.device)
    return out, lse
