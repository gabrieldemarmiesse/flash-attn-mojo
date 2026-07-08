"""Apple-GPU (Metal) forward subpackage: 8x8 simdgroup-matrix kernel.

The kernel (``kernel.mojo``) is at ccv/metal-flash-attention kernel
parity on M1–M4 (see METAL_PLAN.md). This wrapper bridges torch to it.

Bridging: ZERO-COPY over torch's own MPS buffers (the CUDA path's
model). The inputs are laid out head-major fp16 *on the MPS device*
(cast + GQA-expand + transpose + contiguous — all torch MPS ops), the
outputs are allocated as MPS tensors, and the mojo launcher binds every
buffer's Metal GPU virtual address directly (extracted via the
``gpuAddress`` selector — see ``_mps.py``). No host round-trip, no
mojo-owned staging buffers. Before dispatch we revive the tensors'
MTLHeaps and ``torch.mps.synchronize()`` (see ``_mps.revive_heaps``:
mojo doesn't declare foreign buffers to its encoder, and macOS evicts
idle heaps). The only residual copy is the on-device layout transform
the kernel's fixed head-major indexing requires — bandwidth-bound on
unified memory, ~1000x cheaper than the old CPU round-trip. (An earlier
note here claimed zero-copy was impossible; it wasn't — the missing
pieces were the VA extraction and heap revival. The sibling
``causal-conv1d-mojo`` repo runs the same bridge in CI.)

Envelope: fp16/bf16 q/k/v (computed in fp16), head_dim in {64, 128},
seqlen % 128 == 0, MHA or GQA (Hq % Hkv == 0; GQA is expanded to MHA
on-device). fp32 O out; fp32 natural-log LSE.
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

    # Stage ON THE MPS DEVICE (no CPU round-trip): fp16, head-major
    # (B, H, S, D) contiguous — the layout the kernel indexes
    # (head = b*Hq + h, per-head (S, D) slab). GQA is expanded to MHA
    # here (cheap; the kernel stays MHA-only). These are torch MPS ops,
    # so the results live in torch's MPS allocator and we hand the
    # kernel their Metal GPU VAs (zero-copy).
    def _stage(t: torch.Tensor, heads: int) -> torch.Tensor:
        if heads != nheads:
            t = t.repeat_interleave(nheads // heads, dim=2)
        return (
            t.detach()
            .to(dtype=torch.float16)  # stays on q.device (mps)
            .transpose(1, 2)
            .contiguous()
        )

    q_h = _stage(q, nheads)
    k_h = _stage(k, nheads_kv)
    v_h = _stage(v, nheads_kv)
    o_h = torch.empty(
        (batch, nheads, seqlen, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    lse_h = torch.empty(
        (batch, nheads, seqlen), dtype=torch.float32, device=q.device
    )

    from flash_attn_mojo._mps import gpu_address, revive_heaps
    from flash_attn_mojo.fwd_metal._jit import call_fwd_metal

    def _pre_dispatch() -> None:
        # Runs post-JIT-compile, right before the launch: make every
        # argument tensor's MTLHeap resident (macOS evicts idle GPU
        # memory after ~1 s, and mojo doesn't declare foreign buffers
        # to its encoder — see _mps.revive_heaps), then flush torch's
        # queue so the staging writes above land before the kernel reads
        # them (and O/LSE are resident for the kernel's writes).
        revive_heaps(q_h, k_h, v_h, o_h, lse_h)
        torch.mps.synchronize()

    call_fwd_metal(
        (
            gpu_address(q_h),
            gpu_address(k_h),
            gpu_address(v_h),
            gpu_address(o_h),
            gpu_address(lse_h),
            batch,
            seqlen,
            nheads,
            float(softmax_scale),
            _DTYPE_CODE_FP16,
            head_dim,
        ),
        pre_dispatch=_pre_dispatch,
    )
    # Keep the staged tensors alive until the kernel finishes (their
    # MTLBuffers back the VAs the kernel is using; call_fwd_metal
    # synchronizes the mojo queue internally before returning).
    _ = (q_h, k_h, v_h, o_h, lse_h)

    out = (
        o_h.transpose(1, 2)  # (B, S, H, D)
        .contiguous()
        .to(dtype=q.dtype)  # stays on mps
    )
    lse = lse_h  # (B, H, S) fp32, on mps
    return out, lse
