"""`flash_attn_func` — the public forward+backward API.

FlashAttention-4-class Hopper kernels (`fwd_fa4` / `bwd_fa4`),
JIT-compiled on first use. Supported envelope (asserted with clear
errors):

    * bf16 q/k/v, contiguous (B, S, H, D)
    * head_dim == 128
    * seqlen % 128 == 0
    * causal or non-causal (both at FA4 kernel-time parity, fwd
      and bwd); no dropout/window/alibi
    * MHA or GQA (Hq % Hkv == 0)
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
    _batch, seqlen, _nheads, head_dim = q.shape  # noqa: F841
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
    batch, _, nheads_kv, hd_kv = (
        k.shape if k.dim() == 4 else (0, 0, 0, 0)
    )
    if (
        k.dim() != 4
        or v.shape != k.shape
        or k.shape[0] != q.shape[0]
        or k.shape[1] != seqlen
        or hd_kv != head_dim
        or _nheads % max(nheads_kv, 1) != 0
    ):
        raise ValueError(
            "k/v must be (batch, seqlen, nheads_kv, head_dim) with "
            "Hq % Hkv == 0 (MHA or GQA): "
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
        q, k, v: bf16 CUDA tensors, q (batch, seqlen, nheads,
            head_dim), k/v (batch, seqlen, nheads_kv, head_dim) with
            Hq % Hkv == 0 (head_dim=128, seqlen % 128 == 0).
            Non-CUDA tensors run the pure-PyTorch reference instead.
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
        g = q.shape[2] // k.shape[2]
        k_r = k.repeat_interleave(g, dim=2) if g > 1 else k
        v_r = v.repeat_interleave(g, dim=2) if g > 1 else v
        out = flash_attn_ref(
            q, k_r, v_r, softmax_scale=softmax_scale, causal=causal
        )
        if return_lse:
            scores = (
                torch.einsum(
                    "bshd,bthd->bhst", q.float(), k_r.float()
                )
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


def _check_varlen_envelope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    if q.dim() != 3:
        raise ValueError(
            f"varlen q must be packed (total_tokens, nheads, head_dim), "
            f"got {tuple(q.shape)}"
        )
    _total, nheads, head_dim = q.shape
    if q.dtype != torch.bfloat16:
        raise ValueError(f"only bf16 is supported, got {q.dtype}")
    if head_dim != _SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"only head_dim={_SUPPORTED_HEAD_DIM} is supported, got "
            f"{head_dim}"
        )
    if (
        k.dim() != 3
        or v.shape != k.shape
        or k.shape[2] != head_dim
        or nheads % max(k.shape[1], 1) != 0
    ):
        raise ValueError(
            "k/v must be packed (total_tokens, nheads_kv, head_dim) "
            "with Hq % Hkv == 0 (MHA or GQA): "
            f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must share one dtype")
    for name, cu in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if cu.dim() != 1 or cu.dtype != torch.int32:
            raise ValueError(
                f"{name} must be a 1-D int32 tensor, got "
                f"{tuple(cu.shape)} {cu.dtype}"
            )
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must have the same length"
        )


class _FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal):
        from flash_attn_mojo.fwd_fa4 import fa4_varlen_fwd

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = fa4_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale,
            causal=causal,
        )
        ctx.save_for_backward(
            q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        from flash_attn_mojo.bwd_fa4 import bwd_fa4_varlen

        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = bwd_fa4_varlen(
            q, k, v, out, dout.contiguous(), lse, cu_q, cu_k,
            ctx.softmax_scale, causal=ctx.causal,
        )
        return dq, dk, dv, None, None, None, None


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    *,
    return_lse: bool = False,
):
    """Packed variable-length attention via the FA4-class Mojo
    kernels.

    Args:
        q, k, v: bf16 CUDA tensors packed as (total_tokens, nheads,
            head_dim) / (total_tokens, nheads_kv, head_dim);
            head_dim=128. Current envelope: every sequence length a
            multiple of 128, self-attention lengths (cu_seqlens_q ==
            cu_seqlens_k); the backward additionally requires MHA.
            Non-CUDA tensors run the pure-PyTorch reference instead.
        cu_seqlens_q, cu_seqlens_k: (nseq+1,) int32 cumulative
            sequence lengths, starting at 0.
        max_seqlen_q, max_seqlen_k: accepted for flash-attn signature
            compatibility (computed internally when omitted).
        softmax_scale: defaults to head_dim**-0.5.
        causal: causal masking (per sequence).
        return_lse: also return the packed (nheads, total_q) fp32
            natural-log row logsumexp.

    Returns:
        out (total_tokens, nheads, head_dim), or (out, lse).
    """
    _check_varlen_envelope(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # Value-level envelope checks (one tiny D2H copy; the kernel
    # wrappers sync on cu_seqlens anyway). Kept at the API boundary
    # so violations raise clear ValueErrors, not deep AssertionErrors.
    cu_q_host = cu_seqlens_q.detach().cpu()
    cu_k_host = cu_seqlens_k.detach().cpu()
    seqlens_q = cu_q_host[1:] - cu_q_host[:-1]
    if int(cu_q_host[0]) != 0 or int(cu_k_host[0]) != 0:
        raise ValueError("cu_seqlens must start at 0")
    if bool((seqlens_q < 0).any()):
        raise ValueError("cu_seqlens must be non-decreasing")
    if not torch.equal(cu_q_host, cu_k_host):
        raise ValueError(
            "the current varlen envelope is self-attention only: "
            "cu_seqlens_q must equal cu_seqlens_k elementwise"
        )
    if int(cu_q_host[-1]) != q.shape[0]:
        raise ValueError(
            f"cu_seqlens_q[-1] ({int(cu_q_host[-1])}) must equal "
            f"total_tokens ({q.shape[0]})"
        )
    if bool((seqlens_q % _SEQLEN_MULTIPLE != 0).any()):
        raise ValueError(
            "the current varlen envelope needs every sequence length "
            f"to be a multiple of {_SEQLEN_MULTIPLE}, got "
            f"{seqlens_q[:8].tolist()}..."
        )
    max_len = int(seqlens_q.max()) if len(seqlens_q) else 0
    if max_seqlen_q is not None and max_seqlen_q < max_len:
        raise ValueError(
            f"max_seqlen_q ({max_seqlen_q}) is smaller than the "
            f"longest sequence ({max_len})"
        )
    if max_seqlen_k is not None and max_seqlen_k < max_len:
        raise ValueError(
            f"max_seqlen_k ({max_seqlen_k}) is smaller than the "
            f"longest sequence ({max_len})"
        )

    if not q.is_cuda:
        from flash_attn_mojo.reference import flash_attn_varlen_ref

        g = q.shape[1] // k.shape[1]
        k_r = k.repeat_interleave(g, dim=1) if g > 1 else k
        v_r = v.repeat_interleave(g, dim=1) if g > 1 else v
        out = flash_attn_varlen_ref(
            q, k_r, v_r, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=softmax_scale, causal=causal,
        )
        if return_lse:
            cu = cu_seqlens_q.detach().cpu().tolist()
            lse = torch.empty(
                (q.shape[1], q.shape[0]),
                dtype=torch.float32, device=q.device,
            )
            for i in range(len(cu) - 1):
                s, e = cu[i], cu[i + 1]
                scores = (
                    torch.einsum(
                        "shd,thd->hst", q[s:e].float(), k_r[s:e].float()
                    )
                    * softmax_scale
                )
                if causal:
                    tri = torch.ones(
                        e - s, e - s, dtype=torch.bool,
                        device=scores.device,
                    ).triu(1)
                    scores = scores.masked_fill(tri, float("-inf"))
                lse[:, s:e] = torch.logsumexp(scores, dim=-1)
            return out, lse
        return out

    # The varlen backward is MHA-only for now: refuse GQA up front
    # when gradients are live instead of failing inside backward().
    if (
        q.shape[1] != k.shape[1]
        and torch.is_grad_enabled()
        and (q.requires_grad or k.requires_grad or v.requires_grad)
    ):
        raise ValueError(
            "the varlen backward does not support GQA yet — run GQA "
            "varlen inference under torch.no_grad(), or use MHA for "
            "training"
        )

    out, lse = _FlashAttnVarlenFunc.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal
    )
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
