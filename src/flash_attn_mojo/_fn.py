"""`flash_attn_func` — the public forward+backward API.

FlashAttention-4-class Hopper kernels (`fwd_fa4` / `bwd_fa4`),
JIT-compiled on first use. Supported envelope (asserted with clear
errors):

    * bf16 q/k/v, contiguous (B, S, H, D)
    * head_dim == 128
    * seqlen % 128 == 0
    * causal or non-causal (both at FA4 kernel-time parity, fwd
      and bwd); no dropout/window/alibi
    * Hq == Hk (no MQA/GQA)
    * CUDA sm90 (Hopper)

Non-CUDA tensors fall through to `flash_attn_ref` (pure-PyTorch SDPA,
natively differentiable) so the API stays usable for debugging on any
device.

Performance: at the canonical B=2 S=8192 H=16 D=128 both kernels
match Tri Dao's FlashAttention-4 (`flash_attn.cute`) kernel time on
H100 within run-to-run variance. See HANDOFF.md for the numbers.
"""

from __future__ import annotations

import torch

from flash_attn_mojo.reference import flash_attn_ref

_SUPPORTED_HEAD_DIM = 128
_SEQLEN_MULTIPLE = 128


def _check_envelope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> None:
    if q.dim() != 4:
        raise ValueError(
            f"q must be (batch, seqlen, nheads, head_dim), got "
            f"{tuple(q.shape)}"
        )
    _batch, seqlen, _nheads, head_dim = q.shape
    if q.dtype != torch.bfloat16:
        raise ValueError(f"only bf16 is supported, got {q.dtype}")
    if head_dim != _SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"only head_dim={_SUPPORTED_HEAD_DIM} is supported, got "
            f"{head_dim}"
        )
    if seqlen % _SEQLEN_MULTIPLE != 0:
        raise ValueError(
            f"seqlen must be a multiple of {_SEQLEN_MULTIPLE}, got "
            f"{seqlen}"
        )
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "k and v must match q's shape (Hq == Hk; no MQA/GQA): "
            f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must share one dtype")


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, causal):
        from flash_attn_mojo.fwd_fa4 import fa4_fwd

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = fa4_fwd(q, k, v, softmax_scale, causal=causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        from flash_attn_mojo.bwd_fa4 import bwd_fa4

        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = bwd_fa4(
            q, k, v, out, dout.contiguous(), lse, ctx.softmax_scale,
            causal=ctx.causal,
        )
        return dq, dk, dv, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    *,
    return_lse: bool = False,
):
    """Scaled-dot-product attention via the FA4-class Mojo kernels.

    Args:
        q, k, v: (batch, seqlen, nheads, head_dim) bf16 CUDA tensors
            (head_dim=128, seqlen % 128 == 0, Hq == Hk). Non-CUDA
            tensors run the pure-PyTorch reference instead.
        softmax_scale: defaults to head_dim**-0.5.
        causal: causal masking (fully differentiable).
        return_lse: also return the (batch, nheads, seqlen) fp32
            natural-log row logsumexp.

    Returns:
        out, or (out, lse) if return_lse.
    """
    _check_envelope(q, k, v, causal)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    if not q.is_cuda:
        out = flash_attn_ref(
            q, k, v, softmax_scale=softmax_scale, causal=causal
        )
        if return_lse:
            scores = (
                torch.einsum("bshd,bthd->bhst", q.float(), k.float())
                * softmax_scale
            )
            if causal:
                s_q = scores.shape[-2]
                tri = torch.ones(
                    s_q, s_q, dtype=torch.bool, device=scores.device
                ).triu(1)
                scores = scores.masked_fill(tri, float("-inf"))
            return out, torch.logsumexp(scores, dim=-1)
        return out

    out, lse = _FlashAttnFunc.apply(q, k, v, softmax_scale, causal)
    if return_lse:
        return out, lse
    return out


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    *,
    return_lse: bool = False,
):
    """`flash_attn_func` over a packed (B, S, 3, H, D) tensor."""
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q, k, v, softmax_scale, causal, return_lse=return_lse
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    *,
    return_lse: bool = False,
):
    """`flash_attn_func` with k/v packed as (B, S, 2, H, D)."""
    k, v = kv.unbind(dim=2)
    return flash_attn_func(
        q, k, v, softmax_scale, causal, return_lse=return_lse
    )
