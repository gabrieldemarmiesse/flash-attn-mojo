"""`flash_attn_func` — the public forward+backward API.

Mirrors upstream `flash_attn.flash_attn_func` (the v2.x API). The
autograd op dispatches to the GPU kernels (`fwd` + `bwd` subpackages)
when `q.is_cuda`; CPU fallback uses `flash_attn_ref` (pure-PyTorch
SDPA).

STATUS: scaffolding only. The Mojo kernels are stubbed out and raise
`NotImplementedError`. The infrastructure around them (autograd
Function, torch.library.custom_op registration, fake-tensor metadata)
is in place so the kernel work, when added, slots in without further
refactoring.
"""

from __future__ import annotations

import torch

from flash_attn_mojo.reference import flash_attn_ref


# Sentinel for the "no window" case in flash-attn 2's sliding-window
# parameter — `window_size=(-1, -1)` means full attention.
_NO_WINDOW = (-1, -1)


def _fwd_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Forward dispatch. Returns (out, lse, rng_state).

    Current kernel limitations (the simplest viable initial impl):
    only fp16, only head_dim=64, no causal, no dropout, no alibi,
    no softcap, no window, no MQA/GQA. Anything outside that envelope
    raises NotImplementedError so callers see a clear error rather
    than silently-wrong results.
    """
    from flash_attn_mojo.fwd import native_fwd

    # The mha_single_batch port goes through `linalg.matmul.gpu.multistage_mma`,
    # which derives its MMA shape via `get_mma_shape[input, accum]`. For input=bf16
    # the chosen shape is m16n8k16 (one PTX `mma.sync` instruction with the largest
    # K we get on Ampere/Ada); for input=fp16 the published Mojo stdlib at the
    # version we pin (mojo-compiler 1.0.0b1) only ships m16n8k8, so the multi-stage
    # gemm fails to instantiate. We could route fp16 through a hand-rolled m16n8k8
    # gemm but that defeats the point of using `multistage_mma`. So this entry
    # point is bf16-only for now; an fp16 path lands once Mojo gains m16n8k16 fp16
    # in the public stdlib (it's already in MAX's `tensor_core.get_mma_shape`).
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "flash_attn_mojo current kernel supports bf16 only "
            f"(got {q.dtype}). See `_fn.py` for the why."
        )
    if q.shape[-1] not in (32, 64, 128):
        raise NotImplementedError(
            f"flash_attn_mojo current kernel supports head_dim in "
            f"(32, 64, 128) (got {q.shape[-1]})."
        )
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError(
            f"flash_attn_mojo: dropout_p must be in [0, 1), got {dropout_p}."
        )
    nheads_q = q.shape[2]
    nheads_kv = k.shape[2]
    if nheads_q % nheads_kv != 0:
        raise ValueError(
            f"flash_attn_mojo: nheads_q ({nheads_q}) must be a multiple of "
            f"nheads_kv ({nheads_kv}) for MQA/GQA."
        )
    if v.shape[2] != nheads_kv:
        raise ValueError(
            f"flash_attn_mojo: k.shape[2] ({nheads_kv}) must match "
            f"v.shape[2] ({v.shape[2]})."
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    out = torch.empty_like(q)
    # lse: (batch, nheads, seqlen) fp32. Contiguous so the kernel can
    # use simple per-row indexing.
    lse = torch.empty(
        q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32, device=q.device
    )

    window_left, window_right = window_size
    # ALiBi: normalise slopes to fp32 contiguous. None ⇒ pass null ptr +
    # zero strides (the kernel runtime-checks for ptr == 0). 1D
    # (nheads,) ⇒ broadcast across batch with alibi_b_stride = 0. 2D
    # (batch, nheads) ⇒ both strides nonzero.
    alibi_slopes_buf: torch.Tensor | None = None
    if alibi_slopes is not None:
        slopes = alibi_slopes
        if slopes.dtype != torch.float32:
            slopes = slopes.to(torch.float32)
        if not slopes.is_contiguous():
            slopes = slopes.contiguous()
        if slopes.dim() == 1:
            if slopes.shape[0] != nheads_q:
                raise ValueError(
                    f"flash_attn_mojo: alibi_slopes shape (nheads,) expected "
                    f"({nheads_q},), got {tuple(slopes.shape)}."
                )
            alibi_b_stride = 0
            alibi_h_stride = slopes.stride(0)
        elif slopes.dim() == 2:
            if slopes.shape[0] != q.shape[0] or slopes.shape[1] != nheads_q:
                raise ValueError(
                    f"flash_attn_mojo: alibi_slopes shape (batch, nheads) "
                    f"expected ({q.shape[0]}, {nheads_q}), got "
                    f"{tuple(slopes.shape)}."
                )
            alibi_b_stride = slopes.stride(0)
            alibi_h_stride = slopes.stride(1)
        else:
            raise ValueError(
                f"flash_attn_mojo: alibi_slopes must be 1D or 2D, got "
                f"{slopes.dim()}D."
            )
        alibi_slopes_buf = slopes
        alibi_addr = slopes.data_ptr()
    else:
        alibi_addr = 0
        alibi_b_stride = 0
        alibi_h_stride = 0
    # Dropout RNG state. Upstream returns a (seed, offset) uint64 pair as
    # the third element of the `return_attn_probs=True` tuple so the
    # backward can regenerate the mask. We don't have a backward yet, but
    # we still produce a valid pair when dropout is active so the
    # contract holds for future use.
    if dropout_p > 0.0:
        rng_state = torch.empty(2, dtype=torch.int64, device=q.device)
        # Cheap, non-cryptographic seed: a single host-side draw. Good
        # enough for fwd-only training (the kernel uses a fixed-key
        # mixer to expand into per-element bits).
        seed = int(
            torch.randint(0, 2**62, (1,), dtype=torch.int64).item()
        )
        offset = 0
        rng_state[0] = seed
        rng_state[1] = offset
    else:
        rng_state = None
        seed = 0
        offset = 0
    native_fwd(
        q, k, v, out, softmax_scale, causal, nheads_kv, softcap, lse,
        window_left=int(window_left), window_right=int(window_right),
        alibi_addr=int(alibi_addr),
        alibi_b_stride=int(alibi_b_stride),
        alibi_h_stride=int(alibi_h_stride),
        dropout_p=float(dropout_p),
        rng_seed=int(seed),
        rng_offset=int(offset),
    )
    # Keep alibi_slopes_buf alive through the call.
    del alibi_slopes_buf
    return out, lse, rng_state


def _bwd_dispatch(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward dispatch. Returns (dq, dk, dv).

    TODO: replace with the Mojo kernel once `bwd/` is implemented.
    """
    raise NotImplementedError(
        "flash_attn_mojo: GPU backward kernel not yet implemented."
    )


class _FlashAttnFn(torch.autograd.Function):
    """fp16/bf16 autograd op for full (non-varlen) attention.

    Matches upstream's `_flash_attn_func` autograd.Function semantics.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int, int],
        softcap: float,
        alibi_slopes: torch.Tensor | None,
        deterministic: bool,
        return_attn_probs: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        out, lse, rng_state = _fwd_dispatch(
            q, k, v, dropout_p, softmax_scale, causal, window_size,
            softcap, alibi_slopes, deterministic,
        )
        ctx.save_for_backward(q, k, v, out, lse, alibi_slopes)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        if return_attn_probs:
            # Upstream also exposes the softmax denominator and (with
            # dropout) the RNG state. `rng_state` is a 2-element uint64
            # tensor (seed, offset) when dropout is active, None otherwise.
            return out, lse, rng_state
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        dout = grad_outputs[0]
        q, k, v, out, lse, alibi_slopes = ctx.saved_tensors
        dq, dk, dv = _bwd_dispatch(
            dout, q, k, v, out, lse,
            ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            ctx.window_size, ctx.softcap, alibi_slopes, ctx.deterministic,
        )
        # forward arg order: q, k, v, dropout_p, softmax_scale, causal,
        # window_size, softcap, alibi_slopes, deterministic,
        # return_attn_probs. Returns map 1:1 with None for
        # non-differentiable inputs.
        return dq, dk, dv, None, None, None, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = _NO_WINDOW,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Multi-head scaled-dot-product attention with Flash Attention's
    block-tiled algorithm.

    q, k, v: (batch, seqlen, nheads, headdim). Note: nheads_kv may differ
        from nheads_q (multi-query/grouped-query attention) — k and v
        share the same nheads_kv.
    dropout_p: dropout probability on the attention matrix.
    softmax_scale: scale applied before softmax. Defaults to
        `1 / sqrt(headdim)`.
    causal: if True, apply lower-triangular causal mask.
    window_size: `(left, right)` sliding-window mask, both in tokens.
        `(-1, -1)` = no window (the default). With causal=True, only
        the `left` value matters.
    softcap: if > 0, apply `softcap * tanh(scores / softcap)` for
        attention-softcap (Gemma 2 / Grok). 0 disables.
    alibi_slopes: (nheads,) or (batch, nheads) ALiBi slopes.
    deterministic: if True, force the deterministic (slower) backward.
    return_attn_probs: if True, return `(out, softmax_lse, rng_state)`
        — needed for debugging or for stacking attention layers.

    Returns: out of shape (batch, seqlen, nheads, headdim).
    """
    if q.device.type != "cuda":
        # No Mojo CPU kernel yet — fall back to the pure-PyTorch
        # reference for CPU inputs.
        return flash_attn_ref(
            q, k, v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
        )
    result = _FlashAttnFn.apply(
        q, k, v, dropout_p, softmax_scale, causal, window_size,
        softcap, alibi_slopes, deterministic, return_attn_probs,
    )
    return result


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = _NO_WINDOW,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Variable-length / packed-batch attention.

    q: (total_q, nheads_q, head_dim).
    k, v: (total_k, nheads_kv, head_dim).
    cu_seqlens_q, cu_seqlens_k: (batch+1,) int32 prefix sums of the
        per-batch seqlens.
    max_seqlen_q, max_seqlen_k: ints (currently informational; the
        wrapper slices and dispatches per-batch).

    Current limitations of this first-cut implementation:
    - Python-level wrapper: loops over batches on the host, slices Q/K/V,
      calls `flash_attn_func` per slice. Correct but slow; a kernel-side
      varlen path is separate work.
    - Requires `seqlen_q_b == seqlen_k_b` for every batch element b — our
      current kernel doesn't yet handle different Q/K seqlens. Raises
      `NotImplementedError` otherwise.
    - `block_table` (paged KV) is not supported.
    - `return_attn_probs=True` returns the per-batch LSEs concatenated
      along the seqlen axis as `(nheads_q, total_q)` to roughly match
      upstream's shape; `rng_state` is propagated from the last batch
      element when dropout is active (not a faithful varlen RNG).
    """
    if block_table is not None:
        raise NotImplementedError(
            "flash_attn_mojo.flash_attn_varlen_func: block_table (paged KV) "
            "is not supported yet."
        )
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must be 1-D tensors."
        )
    if cu_seqlens_q.shape[0] != cu_seqlens_k.shape[0]:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must have the same length "
            f"(got {cu_seqlens_q.shape[0]} vs {cu_seqlens_k.shape[0]})."
        )
    batch = cu_seqlens_q.shape[0] - 1
    if batch < 1:
        raise ValueError(
            f"cu_seqlens_q implies batch={batch}; need at least 1."
        )

    # Materialise the prefix sums on host once (one D->H sync, not 2*batch).
    cu_q = cu_seqlens_q.detach().to("cpu", dtype=torch.int64).tolist()
    cu_k = cu_seqlens_k.detach().to("cpu", dtype=torch.int64).tolist()

    # Pre-validate per-batch shape compatibility before doing any work.
    for b in range(batch):
        Lq = cu_q[b + 1] - cu_q[b]
        Lk = cu_k[b + 1] - cu_k[b]
        if Lq != Lk:
            raise NotImplementedError(
                "flash_attn_mojo.flash_attn_varlen_func currently requires "
                f"seqlen_q_b == seqlen_k_b for every batch element (batch "
                f"{b}: seqlen_q={Lq}, seqlen_k={Lk}). The underlying kernel "
                "does not yet handle Q/K seqlen mismatch; this is separate "
                "work."
            )

    nheads_q = q.shape[1]
    out = torch.empty_like(q)
    # Per-batch LSE collection (only when return_attn_probs).
    lse_chunks: list[torch.Tensor] = []
    rng_state_last: torch.Tensor | None = None

    for b in range(batch):
        sq, eq = cu_q[b], cu_q[b + 1]
        sk, ek = cu_k[b], cu_k[b + 1]
        if eq == sq:
            # Empty batch element — nothing to do (out slice is already empty).
            continue
        # ALiBi slopes: per-(batch, nheads_q) entries flatten to the b-th row.
        alibi_b: torch.Tensor | None
        if alibi_slopes is None:
            alibi_b = None
        elif alibi_slopes.dim() == 1:
            alibi_b = alibi_slopes
        elif alibi_slopes.dim() == 2:
            alibi_b = alibi_slopes[b]
        else:
            raise ValueError(
                "alibi_slopes must be 1-D (nheads,) or 2-D (batch, nheads)."
            )

        q_b = q[sq:eq].unsqueeze(0)  # (1, L, nheads_q, D)
        k_b = k[sk:ek].unsqueeze(0)  # (1, L, nheads_kv, D)
        v_b = v[sk:ek].unsqueeze(0)
        res = flash_attn_func(
            q_b, k_b, v_b,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_b,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
        if return_attn_probs:
            out_b, lse_b, rng_b = res
            # lse_b: (1, nheads_q, L) -> (nheads_q, L)
            lse_chunks.append(lse_b.squeeze(0))
            if rng_b is not None:
                rng_state_last = rng_b
        else:
            out_b = res
        out[sq:eq] = out_b.squeeze(0)

    if return_attn_probs:
        if lse_chunks:
            lse_full = torch.cat(lse_chunks, dim=-1)  # (nheads_q, total_q)
        else:
            lse_full = torch.empty(
                nheads_q, 0, dtype=torch.float32, device=q.device
            )
        return out, lse_full, rng_state_last
    return out


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = _NO_WINDOW,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Packed-QKV variant of `flash_attn_func`.

    qkv: (batch, seqlen, 3, nheads, head_dim) — Q, K, V stacked along
        dim=2. Unstacked and forwarded to `flash_attn_func`.
    """
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = _NO_WINDOW,
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Packed-KV variant of `flash_attn_func`.

    q: (batch, seqlen_q, nheads_q, head_dim).
    kv: (batch, seqlen_k, 2, nheads_kv, head_dim) — K, V stacked along
        dim=2. Supports MQA/GQA when nheads_q != nheads_kv.
    """
    k, v = kv.unbind(dim=2)
    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
