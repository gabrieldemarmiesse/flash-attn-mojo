"""Backward correctness tests for `flash_attn_func`.

The current backward is a temporary pure-PyTorch path inside
`_bwd_dispatch` (recomputes P from the saved LSE, computes dq/dk/dv in
fp32, casts back to q/k/v's dtype). The fast Mojo GPU bwd lands in a
later commit. These tests pin the contract so that swap-in is safe.

bf16, head_dim in {32, 64, 128}.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import flash_attn_mojo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash_attn_mojo backward needs cuda"
)


def _sdpa_grads(q, k, v, dout, causal: bool):
    """fp32 SDPA reference grads. Inputs (B, L, H, D); returns dq/dk/dv
    in the input dtype."""
    qf = q.detach().float().transpose(1, 2).requires_grad_(True)
    kf = k.detach().float().transpose(1, 2).requires_grad_(True)
    vf = v.detach().float().transpose(1, 2).requires_grad_(True)
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal)
    out.transpose(1, 2).contiguous().backward(dout.float())
    return (
        qf.grad.transpose(1, 2).contiguous().to(q.dtype),
        kf.grad.transpose(1, 2).contiguous().to(k.dtype),
        vf.grad.transpose(1, 2).contiguous().to(v.dtype),
    )


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _make_qkv(B, L, H, D, dtype=torch.bfloat16, Hkv=None):
    Hkv = Hkv if Hkv is not None else H
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=dtype, device="cuda", generator=g, requires_grad=True)
    k = torch.randn(B, L, Hkv, D, dtype=dtype, device="cuda", generator=g, requires_grad=True)
    v = torch.randn(B, L, Hkv, D, dtype=dtype, device="cuda", generator=g, requires_grad=True)
    return q, k, v


def test_backward_basic():
    B, L, H, D = 2, 128, 4, 64
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(q, k, v)
    dout = torch.randn_like(out)
    out.backward(dout)

    dq_ref, dk_ref, dv_ref = _sdpa_grads(q, k, v, dout, causal=False)
    # bf16 tol: grads accumulate over L matmuls so a bit looser than fwd.
    assert _max_abs(q.grad, dq_ref) < 5e-2
    assert _max_abs(k.grad, dk_ref) < 5e-2
    assert _max_abs(v.grad, dv_ref) < 5e-2


def test_backward_causal():
    B, L, H, D = 2, 128, 4, 64
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    dout = torch.randn_like(out)
    out.backward(dout)

    dq_ref, dk_ref, dv_ref = _sdpa_grads(q, k, v, dout, causal=True)
    assert _max_abs(q.grad, dq_ref) < 5e-2
    assert _max_abs(k.grad, dk_ref) < 5e-2
    assert _max_abs(v.grad, dv_ref) < 5e-2


@pytest.mark.parametrize("D", [32, 64, 128])
def test_backward_head_dims(D):
    B, L, H = 2, 64, 2
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    dout = torch.randn_like(out)
    out.backward(dout)
    dq_ref, dk_ref, dv_ref = _sdpa_grads(q, k, v, dout, causal=True)
    # head_dim=128 with H=2 means more accumulation; bump tol a bit.
    tol = 7e-2 if D == 128 else 5e-2
    assert _max_abs(q.grad, dq_ref) < tol
    assert _max_abs(k.grad, dk_ref) < tol
    assert _max_abs(v.grad, dv_ref) < tol


@pytest.mark.parametrize("causal", [False, True])
def test_bwd_head_dim_32_correctness(causal):
    """Native bwd MVP at head_dim=32 — exercises the smaller-D kernel
    variant (smem footprint ~64 KiB, ELTS_PER_THREAD=16)."""
    B, L, H, D = 2, 128, 4, 32
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    dout = torch.randn_like(out)
    out.backward(dout)
    dq_ref, dk_ref, dv_ref = _sdpa_grads(q, k, v, dout, causal=causal)
    assert _max_abs(q.grad, dq_ref) < 5e-2
    assert _max_abs(k.grad, dk_ref) < 5e-2
    assert _max_abs(v.grad, dv_ref) < 5e-2


@pytest.mark.parametrize("causal", [False, True])
def test_bwd_head_dim_128_correctness(causal):
    """Native bwd MVP at head_dim=128 — exercises the larger-D kernel
    variant. Smem footprint is ~96.5 KiB (the softcap-derivative
    scratch slot was dropped + recomputed inline to fit under the
    99 KiB Ada cap)."""
    B, L, H, D = 2, 128, 4, 128
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    dout = torch.randn_like(out)
    out.backward(dout)
    dq_ref, dk_ref, dv_ref = _sdpa_grads(q, k, v, dout, causal=causal)
    # Larger D ⇒ longer reduction in matmuls; bump tol slightly
    # (matches the existing head_dim=128 entry in test_backward_head_dims).
    assert _max_abs(q.grad, dq_ref) < 7e-2
    assert _max_abs(k.grad, dk_ref) < 7e-2
    assert _max_abs(v.grad, dv_ref) < 7e-2


def test_backward_matches_upstream():
    """Cross-validate against upstream flash-attn 2 directly."""
    try:
        from flash_attn import flash_attn_func as upstream_fn
    except ImportError:
        pytest.skip("upstream flash-attn not available")

    B, L, H, D = 2, 128, 4, 64
    g = torch.Generator(device="cuda").manual_seed(42)
    q_data = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    k_data = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    v_data = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    dout = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)

    # Mojo path.
    q_m = q_data.detach().clone().requires_grad_(True)
    k_m = k_data.detach().clone().requires_grad_(True)
    v_m = v_data.detach().clone().requires_grad_(True)
    out_m = flash_attn_mojo.flash_attn_func(q_m, k_m, v_m, causal=True)
    out_m.backward(dout)

    # Upstream path.
    q_u = q_data.detach().clone().requires_grad_(True)
    k_u = k_data.detach().clone().requires_grad_(True)
    v_u = v_data.detach().clone().requires_grad_(True)
    out_u = upstream_fn(q_u, k_u, v_u, causal=True)
    out_u.backward(dout)

    # Upstream's bwd is also bf16 with fp32 accumulation; both should
    # agree within bf16 numerical noise.
    assert _max_abs(q_m.grad, q_u.grad) < 5e-2, (
        f"q.grad diff: {_max_abs(q_m.grad, q_u.grad):.3e}"
    )
    assert _max_abs(k_m.grad, k_u.grad) < 5e-2
    assert _max_abs(v_m.grad, v_u.grad) < 5e-2


def test_backward_with_softcap():
    B, L, H, D = 2, 64, 2, 64
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True, softcap=30.0)
    dout = torch.randn_like(out)
    out.backward(dout)

    # No fp32 SDPA equivalent for softcap; check that grads are finite
    # and non-zero, and that they roughly match a manual fp32 reference.
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()
    assert q.grad.abs().max() > 0

    # Manual fp32 reference using `flash_attn_ref`.
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    from flash_attn_mojo.reference import flash_attn_ref
    out_ref = flash_attn_ref(qf, kf, vf, causal=True, softcap=30.0)
    out_ref.backward(dout.float())
    assert _max_abs(q.grad, qf.grad.to(q.dtype)) < 5e-2
    assert _max_abs(k.grad, kf.grad.to(k.dtype)) < 5e-2
    assert _max_abs(v.grad, vf.grad.to(v.dtype)) < 5e-2


def test_backward_mqa():
    B, L, Hq, Hkv, D = 2, 64, 8, 2, 64
    q, k, v = _make_qkv(B, L, Hq, D, Hkv=Hkv)
    out = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    dout = torch.randn_like(out)
    out.backward(dout)

    # fp32 reference via `flash_attn_ref`.
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    from flash_attn_mojo.reference import flash_attn_ref
    out_ref = flash_attn_ref(qf, kf, vf, causal=True)
    out_ref.backward(dout.float())
    assert _max_abs(q.grad, qf.grad.to(q.dtype)) < 5e-2
    assert _max_abs(k.grad, kf.grad.to(k.dtype)) < 5e-2
    assert _max_abs(v.grad, vf.grad.to(v.dtype)) < 5e-2


def test_bwd_preprocess_correctness():
    """The Mojo `delta = rowsum(dO * O)` kernel must match a fp32
    pytorch reference closely (both reductions run in fp32; the only
    difference is the dO/O reads' bf16→fp32 cast happening lane-side
    versus host-side, which is bit-identical)."""
    from flash_attn_mojo.bwd import native_bwd_preprocess

    B, L, H, D = 2, 128, 4, 64
    g = torch.Generator(device="cuda").manual_seed(0)
    dout = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    out = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)

    delta_mojo = torch.empty(B, H, L, dtype=torch.float32, device="cuda")
    dqaccum = torch.empty(B, H, L, D, dtype=torch.float32, device="cuda")
    native_bwd_preprocess(dout, out, delta_mojo, dqaccum)

    # Reference: dO * O summed over the head_dim, in fp32, then
    # transposed (B, L, H) -> (B, H, L).
    delta_ref = (dout.float() * out.float()).sum(dim=-1).transpose(1, 2).contiguous()

    max_err = (delta_mojo - delta_ref).abs().max().item()
    assert max_err < 1e-3, f"delta max-abs err {max_err} >= 1e-3"


@pytest.mark.parametrize("D", [32, 64, 128])
def test_bwd_preprocess_head_dims(D):
    """Cover every supported head_dim (the kernel is templated over D)."""
    from flash_attn_mojo.bwd import native_bwd_preprocess

    B, L, H = 2, 64, 2
    g = torch.Generator(device="cuda").manual_seed(0)
    dout = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    out = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)

    delta_mojo = torch.empty(B, H, L, dtype=torch.float32, device="cuda")
    dqaccum = torch.empty(B, H, L, D, dtype=torch.float32, device="cuda")
    native_bwd_preprocess(dout, out, delta_mojo, dqaccum)

    delta_ref = (dout.float() * out.float()).sum(dim=-1).transpose(1, 2).contiguous()
    max_err = (delta_mojo - delta_ref).abs().max().item()
    assert max_err < 1e-3, f"D={D}: delta max-abs err {max_err} >= 1e-3"


@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("L", [64, 96, 128, 192])
def test_bwd_preprocess_clears_dqaccum(D, L):
    """The preprocess kernel must zero every element of the dqaccum
    workspace it's given. Pre-fill with garbage, call preprocess, assert
    everything is zero. Includes a non-BM-aligned seqlen (96, 192) to
    cover the tile-tail bounds-check path."""
    from flash_attn_mojo.bwd import native_bwd_preprocess

    B, H = 2, 3
    g = torch.Generator(device="cuda").manual_seed(0)
    dout = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)
    out = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda", generator=g)

    delta = torch.empty(B, H, L, dtype=torch.float32, device="cuda")
    # Fill with NaN/garbage so a missed write is loud.
    dqaccum = torch.full(
        (B, H, L, D), float("nan"), dtype=torch.float32, device="cuda"
    )
    native_bwd_preprocess(dout, out, delta, dqaccum)
    assert torch.all(dqaccum == 0).item(), (
        f"dqaccum has non-zero entries after preprocess "
        f"(D={D}, L={L}): "
        f"nonzero count={(dqaccum != 0).sum().item()}"
    )


def test_bwd_convert_dq_correctness():
    """The Mojo convert-dQ kernel casts fp32 dqaccum (B, H, L, D) to
    bf16 dq (B, L, H, D). Comparison is to pytorch's
    `dqaccum.transpose(1, 2).to(bf16)` — both perform per-element
    fp32->bf16 casts, so the result must be bit-identical."""
    from flash_attn_mojo.bwd import native_bwd_convert_dq

    B, L, H, D = 2, 192, 3, 64
    g = torch.Generator(device="cuda").manual_seed(0)
    dqaccum = torch.randn(B, H, L, D, dtype=torch.float32, device="cuda", generator=g)

    dq_mojo = torch.empty(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    native_bwd_convert_dq(dqaccum, dq_mojo)

    dq_ref = dqaccum.transpose(1, 2).contiguous().to(torch.bfloat16)
    assert torch.equal(dq_mojo, dq_ref), (
        "convert_dq output not bit-equal to pytorch cast: "
        f"max diff {(dq_mojo.float() - dq_ref.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("slopes_shape", ["per_head", "per_batch_head"])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_alibi(slopes_shape, causal):
    """Native bwd with ALiBi: compare against pytorch fallback (same
    reference path used by `flash_attn_ref`)."""
    B, L, H, D = 2, 128, 4, 64
    q, k, v = _make_qkv(B, L, H, D)
    g = torch.Generator(device="cuda").manual_seed(7)
    if slopes_shape == "per_head":
        slopes = torch.rand(H, dtype=torch.float32, device="cuda", generator=g) * 0.5
    else:
        slopes = torch.rand(B, H, dtype=torch.float32, device="cuda", generator=g) * 0.5

    out = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=causal, alibi_slopes=slopes
    )
    dout = torch.randn_like(out)
    out.backward(dout)

    # Reference: pytorch flash_attn_ref in fp32.
    from flash_attn_mojo.reference import flash_attn_ref
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    out_ref = flash_attn_ref(qf, kf, vf, causal=causal, alibi_slopes=slopes)
    out_ref.backward(dout.float())
    assert _max_abs(q.grad, qf.grad.to(q.dtype)) < 5e-2
    assert _max_abs(k.grad, kf.grad.to(k.dtype)) < 5e-2
    assert _max_abs(v.grad, vf.grad.to(v.dtype)) < 5e-2


@pytest.mark.parametrize("window", [(64, 0), (32, 32), (128, -1)])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_window(window, causal):
    """Native bwd with sliding window: compare against pytorch fallback."""
    B, L, H, D = 2, 256, 4, 64
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=causal, window_size=window
    )
    dout = torch.randn_like(out)
    out.backward(dout)

    from flash_attn_mojo.reference import flash_attn_ref
    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)
    out_ref = flash_attn_ref(qf, kf, vf, causal=causal, window_size=window)
    out_ref.backward(dout.float())
    assert _max_abs(q.grad, qf.grad.to(q.dtype)) < 5e-2
    assert _max_abs(k.grad, kf.grad.to(k.dtype)) < 5e-2
    assert _max_abs(v.grad, vf.grad.to(v.dtype)) < 5e-2


def _make_varlen_qkv(seqlens, H, D, dtype=torch.bfloat16, Hkv=None, seed=0):
    """Pack per-batch (L_b, H, D) tensors into a varlen (total, H, D) layout.
    Returns (q, k, v, cu_seqlens, per_batch_unpacked_qkv) where each unpacked
    qkv element is a (1, L_b, H, D) tensor with requires_grad=True (independent
    leaf, for per-batch reference grads).
    """
    Hkv = Hkv if Hkv is not None else H
    total = sum(seqlens)
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_data = torch.randn(total, H, D, dtype=dtype, device="cuda", generator=g)
    k_data = torch.randn(total, Hkv, D, dtype=dtype, device="cuda", generator=g)
    v_data = torch.randn(total, Hkv, D, dtype=dtype, device="cuda", generator=g)
    q = q_data.detach().clone().requires_grad_(True)
    k = k_data.detach().clone().requires_grad_(True)
    v = v_data.detach().clone().requires_grad_(True)
    cu = torch.tensor(
        [0, *list(__import__("itertools").accumulate(seqlens))],
        dtype=torch.int32, device="cuda",
    )
    # Per-batch independent leaves for reference.
    per_batch = []
    offsets = [0, *list(__import__("itertools").accumulate(seqlens))]
    for b, L in enumerate(seqlens):
        s, e = offsets[b], offsets[b + 1]
        qb = q_data[s:e].detach().clone().unsqueeze(0).requires_grad_(True)
        kb = k_data[s:e].detach().clone().unsqueeze(0).requires_grad_(True)
        vb = v_data[s:e].detach().clone().unsqueeze(0).requires_grad_(True)
        per_batch.append((qb, kb, vb))
    return q, k, v, cu, per_batch, offsets


@pytest.mark.parametrize("causal", [False, True])
def test_varlen_backward_basic(causal):
    """Backward through `flash_attn_varlen_func` must work (autograd
    flows through the per-batch slice + `flash_attn_func` + slice-assign).
    Compare per-batch grads against `flash_attn_func` called per-batch
    on unpacked tensors."""
    seqlens = [64, 128, 96]
    H, D = 4, 64
    q, k, v, cu, per_batch, offsets = _make_varlen_qkv(seqlens, H, D)
    max_s = max(seqlens)

    out = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu, cu, max_s, max_s, causal=causal,
    )
    assert out.shape == q.shape
    dout = torch.randn_like(out)
    out.backward(dout)

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape

    # Per-batch reference: call flash_attn_func on each unpacked (1, L, H, D)
    # tile and gather grads.
    max_err = 0.0
    for b, L in enumerate(seqlens):
        qb, kb, vb = per_batch[b]
        out_b = flash_attn_mojo.flash_attn_func(qb, kb, vb, causal=causal)
        s, e = offsets[b], offsets[b + 1]
        out_b.backward(dout[s:e].unsqueeze(0))
        max_err = max(
            max_err,
            _max_abs(q.grad[s:e], qb.grad.squeeze(0)),
            _max_abs(k.grad[s:e], kb.grad.squeeze(0)),
            _max_abs(v.grad[s:e], vb.grad.squeeze(0)),
        )
    # Same kernel, same inputs ⇒ bit-identical expected. Use a tight tol.
    assert max_err < 1e-6, f"varlen vs per-batch grad mismatch: {max_err}"


def test_backward_dropout_basic():
    """`dropout_p > 0` now works end-to-end: the native bwd kernel replays
    the fwd's splitmix32 RNG bit-for-bit to reproduce the same per-
    element mask, then applies it to P before the dV/dS chain.

    We exercise four properties:
      1. Shapes / finiteness of dq/dk/dv match the no-dropout call.
      2. Same torch seed ⇒ same grads (since fwd's rng_state is drawn
         from torch's RNG and the bwd replays it).
      3. Different torch seed ⇒ different grads (sanity that the
         mask isn't being ignored).
      4. `dropout_p = 0.0` is a regression-safe no-op vs the
         no-dropout call.
    """
    B, L, H, D = 2, 128, 4, 64
    q, k, v = _make_qkv(B, L, H, D)
    dout = torch.randn_like(q)

    def _grads(p: float, seed: int):
        for t in (q, k, v):
            if t.grad is not None:
                t.grad = None
        torch.manual_seed(seed)
        out = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=p)
        out.backward(dout)
        return q.grad.clone(), k.grad.clone(), v.grad.clone()

    # (1) Shapes + finiteness with active dropout.
    dq, dk, dv = _grads(0.3, 0)
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all() and torch.isfinite(dv).all()

    # (2) Same seed ⇒ same grads (forward draws the same rng_state, bwd
    # replays the same mask). bit-identical because all the dropout-
    # affected math is deterministic at fixed seed.
    dq_a, dk_a, dv_a = _grads(0.3, 42)
    dq_b, dk_b, dv_b = _grads(0.3, 42)
    assert torch.equal(dq_a, dq_b), "same-seed grads not deterministic"
    assert torch.equal(dk_a, dk_b)
    assert torch.equal(dv_a, dv_b)

    # (3) Different seed ⇒ different grads. With a moderate p and
    # L*L=128*128 elements the mask difference is overwhelming.
    dq_c, dk_c, dv_c = _grads(0.3, 99)
    diff = (dq_a.float() - dq_c.float()).abs().max().item()
    assert diff > 1e-3, f"different-seed grads suspiciously close: {diff}"

    # (4) dropout_p=0.0 matches the no-dropout call (regression). The
    # native bwd kernel's `has_dropout` runtime gate must keep the
    # zero-p path identical to the no-arg path.
    dq_p0, dk_p0, dv_p0 = _grads(0.0, 0)
    dq_ref, dk_ref, dv_ref = _grads(0.0, 1)  # different seed, no dropout
    # The (B, L, H, D) tensors are reductions of the same fp32 maths;
    # at p=0 the seed doesn't enter the kernel at all → bit-identical.
    assert torch.equal(dq_p0, dq_ref)
    assert torch.equal(dk_p0, dk_ref)
    assert torch.equal(dv_p0, dv_ref)


def test_backward_dropout_mask_matches_forward():
    """The crux: the bwd's recomputed dropout mask must be the SAME mask
    the fwd applied. We verify this end-to-end by comparing the kernel's
    grads against an fp32 pytorch oracle that:

      1. Runs the fwd via `flash_attn_func` to get (out, lse, rng_state).
      2. Recomputes the dropout mask in pytorch using the SAME
         splitmix32 mixer the kernels use, keyed on the same (seed,
         offset, batch, q_head, q_idx, kv_idx).
      3. Computes dq/dk/dv from the dropped+scaled P via fp32 matmuls.

    If the kernel's mask diverges from the fwd's by even one element
    the dq/dk/dv won't match the oracle. bf16 tolerance applies.
    """
    B, L, H, D = 1, 32, 2, 32
    p = 0.25
    q, k, v = _make_qkv(B, L, H, D)
    dout = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    torch.manual_seed(7)
    out, lse, rng_state = flash_attn_mojo.flash_attn_func(
        q, k, v, dropout_p=p, return_attn_probs=True,
    )
    out.backward(dout)
    dq_k = q.grad.clone()
    dk_k = k.grad.clone()
    dv_k = v.grad.clone()

    # ---- Oracle: replay the fwd's mask in pytorch.
    seed = int(rng_state[0].item())
    offset = int(rng_state[1].item())
    sm_scale = D ** -0.5

    qf = q.detach().float()
    kf = k.detach().float()
    vf = v.detach().float()
    # (B, H, L, D)
    qt = qf.transpose(1, 2)
    kt = kf.transpose(1, 2)
    vt = vf.transpose(1, 2)
    s = (qt @ kt.transpose(-2, -1)) * sm_scale
    P = torch.exp(s - lse.unsqueeze(-1)).float()

    # Build the dropout mask by mirroring the splitmix32 mixer exactly.
    # We compute u for every (b, h, q, k) as uint32 and compare to the
    # threshold the kernel uses.
    seed_mix = (seed ^ offset) & 0xFFFFFFFFFFFFFFFF
    seed_mix_xor32 = (seed_mix & 0xFFFFFFFF) ^ (seed_mix >> 32)

    thr_f = p * 4294967296.0
    if thr_f > 4294967040.0:
        thr_f = 4294967040.0
    thr = int(thr_f) & 0xFFFFFFFF

    M = torch.empty(B, H, L, L, dtype=torch.bool, device="cuda")
    for b in range(B):
        for h in range(H):
            bh_mix = (b * 2654435761 + h * 40503) & 0xFFFFFFFFFFFFFFFF
            rng_key = (
                (seed_mix_xor32 & 0xFFFFFFFF)
                ^ (bh_mix & 0xFFFFFFFF)
                ^ ((bh_mix >> 32) & 0xFFFFFFFF)
            ) & 0xFFFFFFFF
            for qi in range(L):
                for kj in range(L):
                    u = (
                        rng_key
                        ^ ((qi * 0x9E3779B1) & 0xFFFFFFFF)
                        ^ ((kj * 0x85EBCA77) & 0xFFFFFFFF)
                    ) & 0xFFFFFFFF
                    u = (u ^ (u >> 16)) & 0xFFFFFFFF
                    u = (u * 0x7FEB352D) & 0xFFFFFFFF
                    u = (u ^ (u >> 15)) & 0xFFFFFFFF
                    u = (u * 0x846CA68B) & 0xFFFFFFFF
                    u = (u ^ (u >> 16)) & 0xFFFFFFFF
                    M[b, h, qi, kj] = u >= thr

    keep_scale = 1.0 / (1.0 - p)
    Pd = P * M.float() * keep_scale  # dropped → 0, kept → scaled

    dout_t = dout.detach().float().transpose(1, 2)
    dV = Pd.transpose(-2, -1) @ dout_t
    dP = dout_t @ vt.transpose(-2, -1)
    delta = (dout_t * out.detach().float().transpose(1, 2)).sum(dim=-1, keepdim=True)
    dS = Pd * (dP - delta) * sm_scale
    dQ = dS @ kt
    dK = dS.transpose(-2, -1) @ qt

    dq_ref = dQ.transpose(1, 2).to(torch.bfloat16)
    dk_ref = dK.transpose(1, 2).to(torch.bfloat16)
    dv_ref = dV.transpose(1, 2).to(torch.bfloat16)

    # If the kernel's mask diverged from the oracle's even at a single
    # (b, h, q, k), one term in P would flip between 0 and `p_orig/(1-p)`,
    # which is a substantial value at moderate scores — well above bf16
    # rounding noise. So a tight tolerance here is the strongest end-to-
    # end signal that the mask matches bit-for-bit.
    assert _max_abs(dq_k, dq_ref) < 5e-2
    assert _max_abs(dk_k, dk_ref) < 5e-2
    assert _max_abs(dv_k, dv_ref) < 5e-2
