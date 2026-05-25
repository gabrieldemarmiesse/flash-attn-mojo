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
import torch.nn.functional as F

from flash_attn_mojo.reference import flash_attn_ref


# Sentinel for the "no window" case in flash-attn 2's sliding-window
# parameter — `window_size=(-1, -1)` means full attention.
_NO_WINDOW = (-1, -1)


# Native kernel-supported head_dims. Anything else within upstream's
# envelope (multiple of 8, <= 128) is rounded UP to the nearest of these
# and run with zero-padded q/k/v; the output's padded slots are sliced
# off before return. Upstream does the same (see flash_api.cpp's
# `head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64)`),
# but rounds to {32, 64, 96, 128, 160, 192, 224, 256}. Our supported
# subset is just the powers of two we have kernel variants for.
_NATIVE_HEAD_DIMS = (32, 64, 128)
_MAX_HEAD_DIM = 128


def _round_head_dim(head_dim: int) -> int:
    """Round `head_dim` UP to the nearest natively-supported size.

    Validates against upstream's accepted envelope (multiple of 8,
    <= 256) and rejects sizes we don't have a kernel variant for
    (> 128). Returns the rounded size, which is guaranteed to be in
    `_NATIVE_HEAD_DIMS`.
    """
    if head_dim <= 0 or head_dim % 8 != 0:
        # Upstream wording (csrc/flash_attn/flash_api.cpp):
        #     "head_size should be a multiple of 8".
        raise ValueError(
            f"flash_attn_mojo: head_size should be a multiple of 8, "
            f"got {head_dim}."
        )
    if head_dim > _MAX_HEAD_DIM:
        # Upstream accepts up to 256 by also providing 160/192/224/256
        # kernel variants. We don't yet — see CLAUDE.md ("Kernel-design
        # patterns to mirror") for the work needed to add them.
        raise NotImplementedError(
            f"flash_attn_mojo: head_dim={head_dim} > {_MAX_HEAD_DIM} is not "
            "supported. Native kernel variants for head_dim in "
            "(160, 192, 224, 256) are not yet implemented; see CLAUDE.md "
            "for the kernel-side work to add them."
        )
    for native in _NATIVE_HEAD_DIMS:
        if head_dim <= native:
            return native
    # Unreachable given the <= _MAX_HEAD_DIM check above.
    raise AssertionError(f"unreachable: head_dim={head_dim}")


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
    # gemm but that defeats the point of using `multistage_mma`. A native fp16
    # kernel path lands once Mojo gains m16n8k16 fp16 in the public stdlib (it's
    # already in MAX's `tensor_core.get_mma_shape`); until then, fp16 inputs are
    # handled at the API boundary by `flash_attn_func` (cast q/k/v to bf16, run
    # the bf16 kernel, cast out back to fp16). bf16 has 7-bit mantissa vs fp16's
    # 10-bit so accuracy is slightly worse than a true fp16 path, but bf16's
    # wider 8-bit exponent absorbs the dynamic range of softmax inputs fine.
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(
            "flash_attn_mojo._fwd_dispatch: kernel supports bf16 only "
            f"(got {q.dtype}). fp16 is handled by casting at the API "
            "boundary in flash_attn_func; if you hit this directly you "
            "bypassed that cast."
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


def _bwd_dispatch_native_mvp(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    alibi_slopes: torch.Tensor | None = None,
    window_size: tuple[int, int] = _NO_WINDOW,
    dropout_p: float = 0.0,
    rng_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native MVP backward: bf16 / head_dim in {32, 64, 128} / optional causal / no MQA.

    Allocates delta, dqaccum, dk, dv. Calls native_bwd_preprocess to
    fill delta + zero dqaccum. Calls native_bwd_main to fill dk, dv
    and atomic-add dq contributions into dqaccum. Casts dqaccum (fp32)
    to dq (bf16/fp16) via torch — the dedicated convert_dq kernel
    lands in a subsequent commit.

    fp16 inputs are bf16-cast at the API boundary (same pattern as
    the fwd kernel) since the bwd kernel only specialises on bf16
    for the MVP.

    Dropout is supported: the native bwd kernel replays the fwd's
    splitmix32 dropout RNG bit-for-bit (same seed/offset/mixer) so
    masked positions get P=0 and survivors are scaled by `1/(1-p)`,
    which then propagates correctly through dV/dP/dS into dq/dk/dv.
    `rng_state` is the (seed, offset) int64 cuda tensor returned by
    the forward; when `dropout_p > 0` it must be non-None.
    """
    rng_seed_i: int = 0
    rng_offset_i: int = 0
    if dropout_p > 0.0:
        if rng_state is None:
            raise RuntimeError(
                "flash_attn_mojo backward: dropout_p > 0 requires rng_state "
                "from the forward (the bwd needs the same seed/offset to "
                "replay the fwd's dropout mask)."
            )
        # rng_state is a 2-elt int64 cuda tensor. One D->H sync to read
        # the two scalars — the bwd has many syncs already (delta
        # preprocess, then the main kernel), so this is in the noise.
        rng_host = rng_state.detach().to("cpu", dtype=torch.int64).tolist()
        rng_seed_i = int(rng_host[0])
        rng_offset_i = int(rng_host[1])
    from flash_attn_mojo.bwd import (
        native_bwd_preprocess,
        native_bwd_main,
        native_bwd_convert_dq,
    )

    orig_dtype = q.dtype
    if orig_dtype == torch.float16:
        q_k = q.to(torch.bfloat16)
        k_k = k.to(torch.bfloat16)
        v_k = v.to(torch.bfloat16)
        out_k = out.to(torch.bfloat16)
        dout_k = dout.to(torch.bfloat16)
    else:
        q_k, k_k, v_k, out_k, dout_k = q, k, v, out, dout

    # All inputs must be contiguous and use the canonical (B, L, H, D) /
    # (B, H, L) layout the kernel expects. The autograd wrapper passes
    # contiguous tensors already, but be defensive.
    q_k = q_k.contiguous()
    k_k = k_k.contiguous()
    v_k = v_k.contiguous()
    out_k = out_k.contiguous()
    dout_k = dout_k.contiguous()
    lse_c = lse.contiguous()

    B, L, H, D = q_k.shape
    Hkv = k_k.shape[2]

    # ALiBi: normalise slopes to fp32 contiguous and compute pointer +
    # strides. None ⇒ pass null ptr + zero strides (kernel runtime-
    # checks for ptr == 0). Slopes are indexed per Q-head (not KV-head).
    alibi_buf: torch.Tensor | None = None
    alibi_addr = 0
    alibi_b_stride = 0
    alibi_h_stride = 0
    if alibi_slopes is not None:
        slopes = alibi_slopes
        if slopes.dtype != torch.float32:
            slopes = slopes.to(torch.float32)
        if not slopes.is_contiguous():
            slopes = slopes.contiguous()
        if slopes.dim() == 1:
            alibi_b_stride = 0
            alibi_h_stride = slopes.stride(0)
        elif slopes.dim() == 2:
            alibi_b_stride = slopes.stride(0)
            alibi_h_stride = slopes.stride(1)
        else:
            raise ValueError(
                f"alibi_slopes must be 1D or 2D, got {slopes.dim()}D."
            )
        alibi_buf = slopes
        alibi_addr = slopes.data_ptr()

    delta = torch.empty(B, H, L, dtype=torch.float32, device=q_k.device)
    # Deterministic dqaccum: shape (num_n_blocks, B, H, L, D) fp32.
    # The main bwd kernel writes per-(n_block) slots without atomics
    # (each kv-block-grid block owns its own slot); convert_dq sums
    # across the n_block dim. torch.zeros (= cudaMemsetAsync) replaces
    # the preprocess kernel's old per-slot zeroing pass.
    _BWD_BN = 64  # matches kBwdBlockN; one n_block per BN K-rows.
    num_n_blocks = (L + _BWD_BN - 1) // _BWD_BN
    dqaccum = torch.zeros(
        num_n_blocks, B, H, L, D, dtype=torch.float32, device=q_k.device
    )
    native_bwd_preprocess(dout_k, out_k, delta, dqaccum)

    dk = torch.empty_like(k_k)
    dv = torch.empty_like(v_k)
    native_bwd_main(
        q_k, k_k, v_k, dout_k, lse_c, delta, dk, dv, dqaccum, softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_addr=int(alibi_addr),
        alibi_b_stride=int(alibi_b_stride),
        alibi_h_stride=int(alibi_h_stride),
        window_left=int(window_size[0]),
        window_right=int(window_size[1]),
        dropout_p=float(dropout_p),
        rng_seed=rng_seed_i,
        rng_offset=rng_offset_i,
    )
    del alibi_buf

    # dq: (B, L, H, D) in q_k's dtype. The convert_dq kernel reads
    # dqaccum (B, H, L, D) fp32 and writes dq (B, L, H, D) dtype with
    # the H/L transpose baked in.
    dq = torch.empty_like(q_k)
    native_bwd_convert_dq(dqaccum, dq)
    # The native kernel writes dk/dv directly in the (B, L, H, D) layout.

    if orig_dtype == torch.float16:
        dq = dq.to(torch.float16)
        dk = dk.to(torch.float16)
        dv = dv.to(torch.float16)
    return dq, dk, dv


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
    rng_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward dispatch. Returns (dq, dk, dv).

    TEMPORARY pure-PyTorch implementation. Computes dq/dk/dv from the
    saved (q, k, v, out, lse) and the incoming dout by recomputing the
    attention matrix P from the saved LSE (flash-attn's standard bwd
    trick — using the saved LSE keeps the rounding identical to the
    fwd). All arithmetic runs in fp32; outputs are cast back to q/k/v's
    dtype. Correct but slow — the whole attention matrix is
    materialised in fp32.

    The fast Mojo GPU kernel lands in a subsequent commit; this exists
    so `loss.backward()` actually fills in q.grad / k.grad / v.grad
    end-to-end today, and serves as the correctness oracle for the
    eventual GPU bwd.

    Layout matches `_fwd_dispatch`: q/k/v/out/dout are
    (B, L, H, D); lse is (B, H, L) fp32. Composition of softcap,
    alibi, window, causal exactly mirrors the fwd reference's order
    so the saved LSE matches the recomputed pre-softmax scores.
    """
    # ---- MVP native bwd routing.
    # Inside the MVP envelope (bf16/fp16 cuda, head_dim in {32, 64, 128}, optional
    # causal, MQA/GQA, optional softcap, no alibi/window/dropout,
    # equal seqlen) we call
    # the native bwd kernel for dk/dv/dqaccum and the convert_dq kernel
    # for fp32 -> dtype. Outside the envelope we fall through to the
    # pytorch reference. The MVP envelope expands as feature commits
    # land (causal, MQA, softcap, etc.).
    _in_mvp_envelope = (
        dout.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.shape[-1] in (32, 64, 128)
        and q.shape[2] % k.shape[2] == 0  # MQA/GQA: Hq divisible by Hkv
        and q.shape[1] == k.shape[1]  # equal seqlen
    )
    if _in_mvp_envelope:
        return _bwd_dispatch_native_mvp(
            dout, q, k, v, out, lse, softmax_scale, causal, softcap,
            alibi_slopes=alibi_slopes,
            window_size=window_size,
            dropout_p=dropout_p,
            rng_state=rng_state,
        )

    if dropout_p > 0.0:
        # Faithfully replaying upstream's dropout RNG in PyTorch is
        # nontrivial (kernel-side Philox seed/offset over the (B,H,L_q,L_k)
        # tile grid). Out of scope for this commit; the eventual GPU
        # bwd will handle it natively.
        raise NotImplementedError(
            "flash_attn_mojo backward: dropout_p > 0 is not supported by the "
            "temporary pytorch-fallback backward. Train with dropout_p=0 for "
            "now; the GPU bwd kernel (subsequent commit) will support dropout."
        )

    B, Lq, Hq, D = q.shape
    Lk = k.shape[1]
    Hkv = k.shape[2]

    # (B, L, H, D) -> (B, H, L, D), fp32.
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    out_t = out.transpose(1, 2).float()
    dout_t = dout.transpose(1, 2).float()

    # MQA/GQA: expand kv heads to match q heads.
    repeat = 1
    if Hkv != Hq:
        if Hq % Hkv != 0:
            raise ValueError(
                f"nheads_q ({Hq}) must be a multiple of nheads_kv ({Hkv})."
            )
        repeat = Hq // Hkv
        kt = kt.repeat_interleave(repeat, dim=1)
        vt = vt.repeat_interleave(repeat, dim=1)

    # Pre-softcap scores: qt @ kt^T * softmax_scale. (B, Hq, Lq, Lk)
    scores_raw = torch.matmul(qt, kt.transpose(-2, -1)) * softmax_scale

    if softcap > 0:
        scores = softcap * torch.tanh(scores_raw / softcap)
    else:
        scores = scores_raw

    # ALiBi bias (matches reference.py: -slope * |i - j|, additive).
    if alibi_slopes is not None:
        slopes = alibi_slopes.float()
        i = torch.arange(Lq, device=scores.device).view(1, 1, -1, 1)
        j = torch.arange(Lk, device=scores.device).view(1, 1, 1, -1)
        if slopes.dim() == 1:
            slopes_v = slopes.view(1, -1, 1, 1)
        else:
            slopes_v = slopes.view(B, -1, 1, 1)
        scores = scores + -slopes_v * (i - j).abs().to(scores.dtype)

    # Sliding-window mask.
    if window_size != _NO_WINDOW:
        left, right = window_size
        i = torch.arange(Lq, device=scores.device).view(-1, 1)
        j = torch.arange(Lk, device=scores.device).view(1, -1)
        in_window = (
            ((j >= i - left) | (left < 0))
            & ((j <= i + right) | (right < 0))
        )
        scores = scores.masked_fill(~in_window, float("-inf"))

    # Causal mask.
    if causal:
        mask = torch.ones(Lq, Lk, dtype=torch.bool, device=scores.device).triu(
            Lk - Lq + 1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    # Recompute P from saved LSE: P = exp(scores - lse).
    # Rows fully masked to -inf would give NaNs (-inf - (-inf)); guard
    # by treating those positions as zero post-exp.
    p = torch.exp(scores - lse.unsqueeze(-1))
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    # dV = P^T @ dO
    dvt = torch.matmul(p.transpose(-2, -1), dout_t)  # (B, Hq, Lk, D)
    # dP = dO @ V^T
    dpt = torch.matmul(dout_t, vt.transpose(-2, -1))  # (B, Hq, Lq, Lk)
    # dS_post = P * (dP - rowsum(dO * O))
    # Compute `delta = rowsum(dO * O)` via the on-device Mojo kernel
    # when the tensors live on CUDA (fp32 accumulator, fp32 output).
    # This is the first piece of the bwd path that runs as a native
    # GPU kernel; the rest (P, dV, dK, dQ) still uses pytorch matmuls.
    if dout.is_cuda:
        from flash_attn_mojo.bwd import native_bwd_preprocess

        delta_t = torch.empty(
            dout.shape[0],
            dout.shape[2],
            dout.shape[1],
            dtype=torch.float32,
            device=dout.device,
        )
        # dQaccum workspace — deterministic-slot shape
        # (num_n_blocks, B, H, L, D) fp32 — see the main-bwd path. The
        # pytorch fallback here computes dQ directly so the workspace
        # is allocated but unused; we still pass it through preprocess
        # to keep the same ABI (preprocess no longer zeros it now that
        # we use torch.zeros, but the wrapper still validates shape).
        _BWD_BN = 64
        num_n_blocks = (
            (dout.shape[1] + _BWD_BN - 1) // _BWD_BN
        )
        dqaccum = torch.zeros(
            num_n_blocks,
            dout.shape[0],
            dout.shape[2],
            dout.shape[1],
            dout.shape[3],
            dtype=torch.float32,
            device=dout.device,
        )
        native_bwd_preprocess(dout, out, delta_t, dqaccum)
        # Match the previous keepdim=True shape (B, Hq, Lq, 1) so the
        # broadcast in `dpt - delta` is unchanged.
        delta = delta_t.unsqueeze(-1).float()
    else:
        delta = (dout_t * out_t).sum(dim=-1, keepdim=True)  # (B, Hq, Lq, 1)
    ds_post = p * (dpt - delta)  # gradient wrt post-softcap/post-alibi scores

    # Alibi is purely additive, so d/d(scores_pre_alibi) = d/d(scores_post).
    # Mask positions (causal/window) have p == 0, so ds_post is already 0
    # there — no extra handling needed.

    # Backprop through softcap: d(softcap*tanh(x/softcap))/dx = 1 - tanh(x/softcap)^2
    # We have `scores` (post-softcap, pre-alibi/mask) = softcap * tanh(...),
    # so the local derivative is 1 - (scores_post_softcap / softcap)^2.
    if softcap > 0:
        scores_post_softcap = softcap * torch.tanh(scores_raw / softcap)
        ds_raw = ds_post * (1.0 - (scores_post_softcap / softcap) ** 2)
    else:
        ds_raw = ds_post

    # dQ = dS_raw @ K * softmax_scale; dK = dS_raw^T @ Q * softmax_scale.
    dqt = torch.matmul(ds_raw, kt) * softmax_scale  # (B, Hq, Lq, D)
    dkt = torch.matmul(ds_raw.transpose(-2, -1), qt) * softmax_scale  # (B, Hq, Lk, D)

    # Fold MQA/GQA: sum dk/dv across the q-head groups back to Hkv.
    if repeat != 1:
        dkt = dkt.view(B, Hkv, repeat, Lk, D).sum(dim=2)
        dvt = dvt.view(B, Hkv, repeat, Lk, D).sum(dim=2)

    # (B, H, L, D) -> (B, L, H, D), cast to input dtype.
    dq = dqt.transpose(1, 2).to(q.dtype).contiguous()
    dk = dkt.transpose(1, 2).to(k.dtype).contiguous()
    dv = dvt.transpose(1, 2).to(v.dtype).contiguous()
    return dq, dk, dv


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
        # Save the fwd's RNG state so the bwd can replay the same dropout
        # mask. `rng_state` is a 2-element int64 cuda tensor (seed, offset)
        # when dropout_p > 0, None otherwise. We stash it on `ctx` rather
        # than `save_for_backward` since it's a non-differentiable
        # auxiliary tensor (and pytorch warns on saving such tensors via
        # save_for_backward unless we mark them).
        ctx.rng_state = rng_state
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
            rng_state=ctx.rng_state,
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

    fp16 vs bf16: the Mojo kernel is bf16-only today (mojo-compiler 1.0.0b1
    only ships m16n8k16 for bf16; fp16 m16n8k16 is in MAX but not yet in
    the public stdlib). fp16 inputs are supported at the API boundary by
    casting q/k/v (and alibi_slopes) to bf16, running the bf16 kernel,
    then casting the output back to fp16. Accuracy is slightly worse than
    a true fp16 path (bf16 has 7-bit mantissa vs fp16's 10), but bf16's
    wider 8-bit exponent absorbs softmax dynamic range fine.

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
    # fp16 → bf16 cast-at-API path. See the docstring for the rationale.
    # The kernel itself is bf16-only at the version of mojo we pin. We
    # cast q/k/v (and alibi slopes if present) to bf16, run the kernel,
    # then cast the output back to fp16. LSE stays fp32 (kernel output).
    fp16_in = q.dtype == torch.float16
    if fp16_in:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        # alibi_slopes is normalised to fp32 inside _fwd_dispatch, so no
        # cast is needed here.

    # head_dim round-up. Upstream's flash-attn accepts any head_dim that
    # is a multiple of 8 (up to 256) and runs a kernel sized to the
    # rounded-up head_dim. We mirror that strategy for the head_dims our
    # native kernels don't cover: pad q/k/v's last dim with zeros up to
    # the rounded size, run the kernel, slice the output back. Since
    # the padded q/k slots are zero, Q · K^T over the padded slots
    # contributes 0; and since V's padded slots are zero, P · V also
    # contributes 0 to the output's real slots. The output's padded
    # slots are garbage which we discard. This is a copy (not free) —
    # the proper fix is per-head_dim native kernel variants (see
    # CLAUDE.md). softmax_scale defaults to 1/sqrt(D_user), NOT the
    # rounded D — matches upstream and keeps the attention math sane.
    head_dim_user = q.shape[-1]
    head_dim_rounded = _round_head_dim(head_dim_user)
    if softmax_scale is None:
        # Lock the scale to the USER's head_dim before padding, otherwise
        # _fwd_dispatch's default would use the rounded size.
        softmax_scale = head_dim_user ** -0.5
    if head_dim_rounded != head_dim_user:
        pad = head_dim_rounded - head_dim_user
        # `F.pad` on the last dim produces a contiguous tensor, which the
        # launcher's pad-and-copy path requires (q.stride(1) == head_dim
        # for the seqlen-unaligned chunk).
        q = F.pad(q, (0, pad))
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))
    result = _FlashAttnFn.apply(
        q, k, v, dropout_p, softmax_scale, causal, window_size,
        softcap, alibi_slopes, deterministic, return_attn_probs,
    )
    # Slice the output back to the user's head_dim. The autograd op
    # returns either a Tensor or (out, lse, rng_state); slice only the
    # out tensor — lse is per-row (independent of head_dim).
    def _slice_head(out_t: torch.Tensor) -> torch.Tensor:
        if head_dim_rounded != head_dim_user:
            return out_t[..., :head_dim_user].contiguous()
        return out_t

    if isinstance(result, tuple):
        out, lse, rng_state = result
        out = _slice_head(out)
        if fp16_in:
            out = out.to(torch.float16)
        return out, lse, rng_state
    result = _slice_head(result)
    if fp16_in:
        result = result.to(torch.float16)
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

    Autograd: backward is supported and flows through transparently — the
    per-batch slice + `flash_attn_func` + slice-assign are all
    autograd-traceable, so `out.backward(...)` populates `q.grad`,
    `k.grad`, `v.grad` with the same gradients you'd get by calling
    `flash_attn_func` on each unpacked batch element and gathering.

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


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    cache_seqlens: int | torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = _NO_WINDOW,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: torch.Tensor | None = None,
    num_splits: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """KV-cache attention.

    q: (batch, seqlen_q, nheads_q, head_dim).
    k_cache, v_cache: (batch, seqlen_cache, nheads_kv, head_dim). When
        `k` / `v` are provided they are appended into these tensors at
        offset `cache_seqlens[b]` (in place), and `cache_seqlens` is
        incremented by `seqlen_new` in place (matching upstream).
    k, v: (batch, seqlen_new, nheads_kv, head_dim) — new tokens to
        append to the cache. Optional.
    cache_seqlens: int or (batch,) int32 tensor — current valid length
        per batch. None ⇒ full cache (`seqlen_cache`).

    Current limitations (first-cut python wrapper):
    - The underlying mojo fwd kernel requires `seqlen_q == seqlen_k`.
      So this entry point only supports the case where, after the
      optional append, every batch element has its used K/V length
      exactly equal to `seqlen_q`. The typical autoregressive decode
      shape (`seqlen_q=1`, `seqlen_k=large`) raises NotImplementedError.
    - Not supported (raise NotImplementedError):
      rotary_cos / rotary_sin, cache_batch_idx, cache_leftpad,
      block_table, num_splits != 0.
    """
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError(
            "flash_attn_mojo.flash_attn_with_kvcache: rotary embeddings "
            "(rotary_cos / rotary_sin) are not supported yet."
        )
    if cache_batch_idx is not None:
        raise NotImplementedError(
            "flash_attn_mojo.flash_attn_with_kvcache: cache_batch_idx is "
            "not supported yet."
        )
    if cache_leftpad is not None:
        raise NotImplementedError(
            "flash_attn_mojo.flash_attn_with_kvcache: cache_leftpad is "
            "not supported yet."
        )
    if block_table is not None:
        raise NotImplementedError(
            "flash_attn_mojo.flash_attn_with_kvcache: block_table (paged "
            "KV) is not supported yet."
        )
    if num_splits not in (0, 1):
        raise NotImplementedError(
            "flash_attn_mojo.flash_attn_with_kvcache: num_splits != 0/1 "
            f"(got {num_splits}) is not supported."
        )

    if q.dim() != 4 or k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError(
            "q, k_cache, v_cache must be 4-D "
            "(batch, seqlen, nheads, head_dim)."
        )

    batch, seqlen_q, _nheads_q, _D = q.shape
    seqlen_cache = k_cache.shape[1]
    if v_cache.shape[1] != seqlen_cache:
        raise ValueError(
            "k_cache and v_cache must share their seqlen dim "
            f"(got {k_cache.shape[1]} vs {v_cache.shape[1]})."
        )
    if k_cache.shape[0] != batch or v_cache.shape[0] != batch:
        raise ValueError(
            "q, k_cache, v_cache must share batch dim "
            f"(got {q.shape[0]}, {k_cache.shape[0]}, {v_cache.shape[0]})."
        )

    # Normalise cache_seqlens to a (batch,) int64 host list for the
    # per-batch loop; keep the original tensor so we can update it
    # in place when we append.
    cache_seqlens_t: torch.Tensor | None
    if cache_seqlens is None:
        cache_seqlens_host = [seqlen_cache] * batch
        cache_seqlens_t = None
    elif isinstance(cache_seqlens, int):
        cache_seqlens_host = [int(cache_seqlens)] * batch
        cache_seqlens_t = None
    else:
        if cache_seqlens.dim() != 1 or cache_seqlens.shape[0] != batch:
            raise ValueError(
                f"cache_seqlens must be a (batch,) tensor; got shape "
                f"{tuple(cache_seqlens.shape)} for batch={batch}."
            )
        cache_seqlens_t = cache_seqlens
        cache_seqlens_host = cache_seqlens.detach().to(
            "cpu", dtype=torch.int64
        ).tolist()

    # Validate append shapes and figure out the post-append used length
    # per batch.
    if (k is None) != (v is None):
        raise ValueError("k and v must both be provided, or both None.")
    if k is not None:
        assert v is not None
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError("k, v must be 4-D (batch, seqlen_new, nheads_kv, head_dim).")
        if k.shape[0] != batch or v.shape[0] != batch:
            raise ValueError(
                "k, v batch dim must match q "
                f"(got {k.shape[0]}, {v.shape[0]}, expected {batch})."
            )
        seqlen_new = k.shape[1]
        if v.shape[1] != seqlen_new:
            raise ValueError(
                f"k and v must share seqlen_new (got {k.shape[1]} vs "
                f"{v.shape[1]})."
            )
        if k.shape[2] != k_cache.shape[2] or v.shape[2] != v_cache.shape[2]:
            raise ValueError(
                "k/v nheads_kv must match k_cache/v_cache."
            )
    else:
        seqlen_new = 0

    used_lengths = [cache_seqlens_host[b] + seqlen_new for b in range(batch)]

    # Mojo fwd kernel needs seqlen_q == seqlen_k. Reject the decode
    # shape (seqlen_q < used_length) up front so users see a clear error
    # instead of a confusing kernel failure.
    for b in range(batch):
        if used_lengths[b] != seqlen_q:
            raise NotImplementedError(
                "flash_attn_mojo.flash_attn_with_kvcache currently requires "
                f"seqlen_q == used_kv_length per batch (batch {b}: "
                f"seqlen_q={seqlen_q}, used_kv={used_lengths[b]}). The "
                "underlying mojo fwd kernel does not yet handle "
                "seqlen_q != seqlen_k (the autoregressive decode shape); "
                "this is separate work."
            )
        if used_lengths[b] > seqlen_cache:
            raise ValueError(
                f"cache overflow on batch {b}: cache_seqlens={cache_seqlens_host[b]}"
                f" + seqlen_new={seqlen_new} > seqlen_cache={seqlen_cache}."
            )

    # Append new k/v into the cache slots and update cache_seqlens.
    if k is not None:
        assert v is not None
        for b in range(batch):
            start = cache_seqlens_host[b]
            end = start + seqlen_new
            k_cache[b, start:end].copy_(k[b])
            v_cache[b, start:end].copy_(v[b])
        if cache_seqlens_t is not None:
            cache_seqlens_t.add_(seqlen_new)

    # Per-batch attention. Slices are non-contiguous along seqlen
    # because k_cache[b, :used] is a view into a (B, L, H, D) tensor;
    # the inner (nheads, head_dim) is still contiguous which is what
    # the kernel cares about. We unsqueeze the batch dim for each call.
    out = torch.empty_like(q)
    for b in range(batch):
        used = used_lengths[b]
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

        q_b = q[b : b + 1]                            # (1, seqlen_q, Hq, D)
        k_b = k_cache[b : b + 1, :used].contiguous()  # (1, used, Hkv, D)
        v_b = v_cache[b : b + 1, :used].contiguous()
        out_b = flash_attn_func(
            q_b, k_b, v_b,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_b,
            deterministic=False,
            return_attn_probs=False,
        )
        out[b : b + 1] = out_b

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
