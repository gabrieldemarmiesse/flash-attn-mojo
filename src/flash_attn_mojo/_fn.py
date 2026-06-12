"""`flash_attn_func` — the public forward+backward API.

FlashAttention-4-class Hopper kernels (`fwd_fa4` / `bwd_fa4`),
JIT-compiled on first use. Supported envelope (asserted with clear
errors):

    * bf16 or fp16 q/k/v, contiguous (B, S, H, D)
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

_SUPPORTED_HEAD_DIMS = (64, 128)
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
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"only bf16 and fp16 are supported, got {q.dtype}"
        )
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"only head_dim in {_SUPPORTED_HEAD_DIMS} is supported, "
            f"got {head_dim}"
        )
    if head_dim == 64 and q.dtype == torch.float16:
        raise ValueError(
            "head_dim=64 is bf16-only for now (fp16 needs the n=64 "
            "RS wgmma arm)"
        )
    if seqlen % _SEQLEN_MULTIPLE != 0 and head_dim != 128:
        raise ValueError(
            f"seqlen must be a multiple of {_SEQLEN_MULTIPLE} at "
            f"head_dim={head_dim} (arbitrary seqlens route through "
            f"the varlen path, which is head_dim=128-only), got "
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
    def forward(
        ctx, q, k, v, softmax_scale, causal, window_left=0,
        softcap=0.0,
    ):
        from flash_attn_mojo.fwd_fa4 import fa4_fwd

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = fa4_fwd(
            q, k, v, softmax_scale, causal=causal,
            window_left=window_left, softcap=softcap,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_left = window_left
        ctx.softcap = softcap
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        from flash_attn_mojo.bwd_fa4 import bwd_fa4

        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = bwd_fa4(
            q, k, v, out, dout.contiguous(), lse, ctx.softmax_scale,
            causal=ctx.causal, window_left=ctx.window_left,
            softcap=ctx.softcap,
        )
        return dq, dk, dv, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    *,
    return_lse: bool = False,
):
    """Scaled-dot-product attention via the FA4-class Mojo kernels.

    Args:
        q, k, v: bf16/fp16 CUDA tensors, q (batch, seqlen, nheads,
            head_dim), k/v (batch, seqlen, nheads_kv, head_dim) with
            Hq % Hkv == 0. head_dim 64 or 128; any seqlen at
            head_dim=128 (non-multiples of 128 route through the
            varlen kernels internally), seqlen % 128 == 0 at 64.
            Non-CUDA tensors run the pure-PyTorch reference instead.
        softmax_scale: defaults to head_dim**-0.5.
        causal: causal masking (fully differentiable).
        window_size: (left, 0) with causal=True enables Mistral-style
            sliding-window attention (fully differentiable). v1
            envelope: left % 128 == 0, head_dim=128, seqlen % 128 == 0.
        softcap: Gemma-2 attention-logit softcap, S := softcap *
            tanh(S / softcap) (fully differentiable; composes with
            causal/window/GQA and any seqlen — non-%128 routes
            through varlen). v1 envelope: head_dim=128. The cap
            compiles into the kernel (one JIT variant per value).
        return_lse: also return the (batch, nheads, seqlen) fp32
            natural-log row logsumexp.

    Returns:
        out, or (out, lse) if return_lse.
    """
    _check_envelope(q, k, v, causal)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    window_left = 0
    if window_size != (-1, -1):
        left, right = window_size
        if not causal or right not in (0, -1):
            raise ValueError(
                "window_size currently requires causal=True with "
                "right window 0 (Mistral-style SWA)"
            )
        if left is None or left < 0:
            raise ValueError("window_size left must be >= 0")
        if left % 128 != 0 or q.shape[-1] != 128 or q.shape[1] % 128:
            raise ValueError(
                "window v1 envelope: left % 128 == 0, head_dim=128, "
                "seqlen % 128 == 0"
            )
        window_left = int(left)

    softcap = float(softcap)
    if softcap:
        if softcap < 0:
            raise ValueError("softcap must be >= 0")
        if q.shape[-1] != 128:
            raise ValueError("softcap v1 envelope: head_dim=128")

    if not q.is_cuda:
        # flash_attn_ref handles GQA (repeat-interleave) and the
        # fp32 LSE internally.
        return flash_attn_ref(
            q, k, v, softmax_scale=softmax_scale, causal=causal,
            window_size=(
                (window_left, 0) if window_left else (-1, -1)
            ),
            softcap=softcap,
            return_lse=return_lse,
        )

    batch, seqlen, nheads, head_dim = q.shape
    if seqlen % _SEQLEN_MULTIPLE != 0:
        # Arbitrary seqlen: route through the varlen kernels (one
        # sequence per batch row) — the ragged-tail machinery masks
        # and clamps everything; semantics are identical for
        # self-attention. The %128 fast path below is untouched.
        nheads_kv = k.shape[2]
        cu = torch.arange(
            0, (batch + 1) * seqlen, seqlen,
            dtype=torch.int32, device=q.device,
        )
        out_p, lse_p = _FlashAttnVarlenFunc.apply(
            q.reshape(batch * seqlen, nheads, head_dim),
            k.reshape(batch * seqlen, nheads_kv, head_dim),
            v.reshape(batch * seqlen, nheads_kv, head_dim),
            cu, cu, softmax_scale, causal, None, None, softcap,
        )
        out = out_p.view(batch, seqlen, nheads, head_dim)
        if return_lse:
            # packed (H, B*S) -> dense (B, H, S)
            return out, (
                lse_p.view(nheads, batch, seqlen)
                .transpose(0, 1)
                .contiguous()
            )
        return out

    out, lse = _FlashAttnFunc.apply(
        q, k, v, softmax_scale, causal, window_left, softcap
    )
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
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"only bf16 and fp16 are supported, got {q.dtype}"
        )
    if head_dim != 128:
        raise ValueError(
            "varlen currently supports head_dim=128 only (hdim64 "
            "varlen is a follow-up)"
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
    def forward(
        ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale,
        causal, seqused_q=None, seqused_k=None, softcap=0.0,
    ):
        from flash_attn_mojo.fwd_fa4 import fa4_varlen_fwd

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = fa4_varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale,
            causal=causal, seqused_q=seqused_q, seqused_k=seqused_k,
            softcap=softcap,
        )
        ctx.save_for_backward(
            q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.seqused = (seqused_q, seqused_k)
        ctx.softcap = softcap
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        from flash_attn_mojo.bwd_fa4 import bwd_fa4_varlen

        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        seqused_q, seqused_k = ctx.seqused
        dq, dk, dv = bwd_fa4_varlen(
            q, k, v, out, dout.contiguous(), lse, cu_q, cu_k,
            ctx.softmax_scale, causal=ctx.causal,
            seqused_q=seqused_q, seqused_k=seqused_k,
            softcap=ctx.softcap,
        )
        return (
            dq, dk, dv, None, None, None, None, None, None, None
        )


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
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    softcap: float = 0.0,
    *,
    return_lse: bool = False,
):
    """Packed variable-length attention via the FA4-class Mojo
    kernels.

    Args:
        q, k, v: bf16 CUDA tensors packed as (total_tokens, nheads,
            head_dim) / (total_tokens, nheads_kv, head_dim);
            head_dim=128. Current envelope: arbitrary sequence
            lengths >= 1, self- OR cross-attention (causal cross
            uses the bottom-right diagonal and requires seqlen_q <=
            seqlen_k per sequence), MHA or GQA (fully
            differentiable). Non-CUDA tensors run the pure-PyTorch
            reference instead.
        cu_seqlens_q, cu_seqlens_k: (nseq+1,) int32 cumulative
            sequence lengths, starting at 0.
        max_seqlen_q, max_seqlen_k: accepted for flash-attn signature
            compatibility (computed internally when omitted).
        softmax_scale: defaults to head_dim**-0.5.
        causal: causal masking (per sequence).
        seqused_q, seqused_k: optional (nseq,) int32 tensors — use
            only the first seqused tokens of each sequence (KV-cache
            style over-allocated buffers; cu_seqlens still define
            the memory layout). Unused rows get out/lse/grads of 0.
        softcap: Gemma-2 attention-logit softcap (fully
            differentiable; the cap compiles into the kernel).
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
    mem_lens_q = cu_q_host[1:] - cu_q_host[:-1]
    mem_lens_k = cu_k_host[1:] - cu_k_host[:-1]
    for name, su, lens in (
        ("seqused_q", seqused_q, mem_lens_q),
        ("seqused_k", seqused_k, mem_lens_k),
    ):
        if su is None:
            continue
        if su.dtype != torch.int32 or tuple(su.shape) != (len(lens),):
            raise ValueError(f"{name} must be (nseq,) int32")
        suh = su.detach().cpu()
        if bool((suh < 1).any()) or bool((suh > lens).any()):
            raise ValueError(
                f"{name} must satisfy 1 <= {name}[i] <= seqlen[i]"
            )
    # Effective (used) lengths drive the mask-shape checks below.
    seqlens_q = (
        seqused_q.detach().cpu() if seqused_q is not None
        else mem_lens_q
    )
    seqlens_k = (
        seqused_k.detach().cpu() if seqused_k is not None
        else mem_lens_k
    )
    if int(cu_q_host[0]) != 0 or int(cu_k_host[0]) != 0:
        raise ValueError("cu_seqlens must start at 0")
    if bool((mem_lens_q < 0).any()) or bool((mem_lens_k < 0).any()):
        raise ValueError("cu_seqlens must be non-decreasing")
    if causal and bool((seqlens_q > seqlens_k).any()):
        raise ValueError(
            "causal varlen cross-attention requires seqlen_q <= "
            "seqlen_k for every sequence (bottom-right diagonal; "
            "seqlen_q > seqlen_k would leave query rows attending "
            "nothing)"
        )
    if int(cu_q_host[-1]) != q.shape[0]:
        raise ValueError(
            f"cu_seqlens_q[-1] ({int(cu_q_host[-1])}) must equal "
            f"q total_tokens ({q.shape[0]})"
        )
    if int(cu_k_host[-1]) != k.shape[0]:
        raise ValueError(
            f"cu_seqlens_k[-1] ({int(cu_k_host[-1])}) must equal "
            f"k total_tokens ({k.shape[0]})"
        )
    if bool((mem_lens_q == 0).any()) or bool((mem_lens_k == 0).any()):
        raise ValueError(
            "empty (zero-length) sequences are not supported — drop "
            "them from cu_seqlens"
        )
    max_len_q = int(seqlens_q.max()) if len(seqlens_q) else 0
    max_len_k = int(seqlens_k.max()) if len(seqlens_k) else 0
    if max_seqlen_q is not None and max_seqlen_q < max_len_q:
        raise ValueError(
            f"max_seqlen_q ({max_seqlen_q}) is smaller than the "
            f"longest sequence ({max_len_q})"
        )
    if max_seqlen_k is not None and max_seqlen_k < max_len_k:
        raise ValueError(
            f"max_seqlen_k ({max_seqlen_k}) is smaller than the "
            f"longest sequence ({max_len_k})"
        )

    softcap = float(softcap)
    if softcap < 0:
        raise ValueError("softcap must be >= 0")

    if not q.is_cuda:
        if seqused_q is not None or seqused_k is not None:
            raise NotImplementedError(
                "seqused_{q,k} requires CUDA tensors (the CPU "
                "reference path does not support it)"
            )
        from flash_attn_mojo.reference import flash_attn_varlen_ref

        # flash_attn_varlen_ref handles GQA and the packed
        # (nheads, total_q) fp32 LSE internally.
        return flash_attn_varlen_ref(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            softmax_scale=softmax_scale, causal=causal,
            softcap=softcap, return_lse=return_lse,
        )

    out, lse = _FlashAttnVarlenFunc.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal,
        seqused_q, seqused_k, softcap,
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
