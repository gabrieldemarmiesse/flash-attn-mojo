"""Pure-PyTorch reference for the public flash-attn API.

Used as the CPU fallback (no Mojo CPU kernel yet) and as the ground
truth for the test suite's correctness checks. Mirrors upstream's
`flash_attn.flash_attn_ref` where one exists; otherwise just calls
through to `torch.nn.functional.scaled_dot_product_attention` after
the layout transposes the flash-attn API expects.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def flash_attn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference SDPA in the flash-attn 2.x API conventions.

    Inputs are (batch, seqlen, nheads, headdim) — flash-attn's
    convention. PyTorch's SDPA wants (batch, nheads, seqlen, headdim),
    so we transpose at the boundary.

    Args mirror `flash_attn_func` exactly. With ``return_lse`` the
    result is ``(out, lse)`` where lse is the (batch, nheads, seqlen)
    fp32 natural-log row logsumexp (computed in fp32 from the same
    masked scores regardless of the input dtype — the test suite's
    ground truth).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # (B, L, H, D) → (B, H, L, D)
    q_h = q.transpose(1, 2)
    k_h = k.transpose(1, 2)
    v_h = v.transpose(1, 2)

    # Handle MQA / GQA: if nheads_kv != nheads_q, repeat-interleave
    # k and v along the head axis. PyTorch's SDPA expects matched H.
    n_h_q = q_h.shape[1]
    n_h_kv = k_h.shape[1]
    if n_h_kv != n_h_q:
        if n_h_q % n_h_kv != 0:
            raise ValueError(
                f"nheads_q ({n_h_q}) must be divisible by nheads_kv ({n_h_kv})"
            )
        repeat = n_h_q // n_h_kv
        k_h = k_h.repeat_interleave(repeat, dim=1)
        v_h = v_h.repeat_interleave(repeat, dim=1)

    if (
        alibi_slopes is not None
        or softcap > 0
        or window_size != (-1, -1)
        or (causal and q_h.shape[2] != k_h.shape[2])
    ):
        # These features aren't supported by torch SDPA — do the
        # matmul by hand. (Cross-length causal too: SDPA's
        # is_causal is TOP-LEFT aligned for Lq != Lk, but flash-attn
        # semantics — and the LSE below — use the BOTTOM-RIGHT
        # diagonal triu(Lk - Lq + 1).)
        scores = torch.matmul(q_h * softmax_scale, k_h.transpose(-1, -2))

        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)

        if alibi_slopes is not None:
            # alibi_slopes: (H,) or (B, H). Bias_{i,j} = -slope * (i - j)
            # for j <= i (and -inf beyond, but causal handles that).
            B, H, Lq, Lk = scores.shape
            i = torch.arange(Lq, device=scores.device).view(1, 1, -1, 1)
            j = torch.arange(Lk, device=scores.device).view(1, 1, 1, -1)
            slopes = alibi_slopes.to(scores.dtype)
            if slopes.dim() == 1:
                slopes = slopes.view(1, -1, 1, 1)
            else:
                slopes = slopes.view(B, -1, 1, 1)
            scores = scores + -slopes * (i - j).abs().to(scores.dtype)

        if window_size != (-1, -1):
            left, right = window_size
            B, H, Lq, Lk = scores.shape
            i = torch.arange(Lq, device=scores.device).view(-1, 1)
            j = torch.arange(Lk, device=scores.device).view(1, -1)
            # Keep j in [i - left, i + right]. Use very negative bias
            # instead of -inf so softmax gradients don't explode.
            in_window = (
                ((j >= i - left) | (left < 0))
                & ((j <= i + right) | (right < 0))
            )
            mask = ~in_window
            scores = scores.masked_fill(mask, float("-inf"))

        if causal:
            B, H, Lq, Lk = scores.shape
            mask = torch.ones(Lq, Lk, dtype=torch.bool, device=scores.device).triu(
                Lk - Lq + 1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = scores.softmax(dim=-1)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        out_h = torch.matmul(attn, v_h)
    else:
        # Fast path — PyTorch's fused SDPA handles causal + dropout.
        out_h = F.scaled_dot_product_attention(
            q_h, k_h, v_h,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )

    # (B, H, L, D) → (B, L, H, D)
    out = out_h.transpose(1, 2).contiguous()
    if not return_lse:
        return out

    if dropout_p > 0:
        raise ValueError("return_lse is ill-defined with dropout")
    # fp32 LSE from the same transformed/masked scores.
    scores = torch.matmul(
        q_h.float() * softmax_scale, k_h.float().transpose(-1, -2)
    )
    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    if alibi_slopes is not None:
        B, H, Lq, Lk = scores.shape
        i = torch.arange(Lq, device=scores.device).view(1, 1, -1, 1)
        j = torch.arange(Lk, device=scores.device).view(1, 1, 1, -1)
        slopes = alibi_slopes.float()
        if slopes.dim() == 1:
            slopes = slopes.view(1, -1, 1, 1)
        else:
            slopes = slopes.view(B, -1, 1, 1)
        scores = scores + -slopes * (i - j).abs().float()
    if window_size != (-1, -1):
        left, right = window_size
        Lq, Lk = scores.shape[-2:]
        i = torch.arange(Lq, device=scores.device).view(-1, 1)
        j = torch.arange(Lk, device=scores.device).view(1, -1)
        in_window = (
            ((j >= i - left) | (left < 0))
            & ((j <= i + right) | (right < 0))
        )
        scores = scores.masked_fill(~in_window, float("-inf"))
    if causal:
        Lq, Lk = scores.shape[-2:]
        mask = torch.ones(
            Lq, Lk, dtype=torch.bool, device=scores.device
        ).triu(Lk - Lq + 1)
        scores = scores.masked_fill(mask, float("-inf"))
    return out, torch.logsumexp(scores, dim=-1)


def flash_attn_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size: tuple[int, int] = (-1, -1),
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Packed-varlen reference: per-sequence `flash_attn_ref` over
    (total_tokens, H, D) inputs sliced by cu_seqlens. Differentiable
    (plain torch ops). With ``return_lse`` also returns the packed
    (nheads, total_q) fp32 LSE (FA4's varlen layout)."""
    cu_q = cu_seqlens_q.detach().cpu().tolist()
    cu_k = cu_seqlens_k.detach().cpu().tolist()
    outs = []
    lses = []
    for i in range(len(cu_q) - 1):
        r = flash_attn_ref(
            q[cu_q[i] : cu_q[i + 1]].unsqueeze(0),
            k[cu_k[i] : cu_k[i + 1]].unsqueeze(0),
            v[cu_k[i] : cu_k[i + 1]].unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            return_lse=return_lse,
        )
        if return_lse:
            outs.append(r[0][0])
            lses.append(r[1][0])
        else:
            outs.append(r[0])
    out = torch.cat(outs, dim=0)
    if return_lse:
        return out, torch.cat(lses, dim=1)
    return out
