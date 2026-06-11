"""FA4-kernel correctness: fwd out/lse and end-to-end autograd grads
vs the fp32 SDPA reference, across tail (S % 80 != 0) and exact-fit
seqlens. bf16 noise-floor tolerances."""

from __future__ import annotations

import pytest
import torch

from flash_attn_mojo import flash_attn_func

from conftest import requires_cuda

# 640 = 8*80 exercises the bwd tile_m=80 exact-fit path; the others
# leave a partial tail m-tile. 128 is the minimum supported seqlen.
SEQLENS = [128, 256, 640, 1024]

FWD_TOL = 5e-3
LSE_TOL = 1e-5
BWD_TOL = 2e-2


def _make(seqlen, requires_grad=False):
    q = torch.randn(
        2, seqlen, 4, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=requires_grad,
    )
    k = torch.randn_like(q, requires_grad=requires_grad)
    v = torch.randn_like(q, requires_grad=requires_grad)
    return q, k, v


def _sdpa_fp32(q, k, v):
    import torch.nn.functional as F

    return (
        F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
        )
        .transpose(1, 2)
    )


@requires_cuda
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_fwd_out_and_lse(seqlen):
    q, k, v = _make(seqlen)
    out, lse = flash_attn_func(q, k, v, return_lse=True)
    ref = _sdpa_fp32(q, k, v)
    assert (out.float() - ref).abs().max().item() < FWD_TOL

    scale = q.shape[-1] ** -0.5
    ref_lse = torch.logsumexp(
        torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * scale,
        dim=-1,
    )
    assert (lse - ref_lse).abs().max().item() < LSE_TOL


@requires_cuda
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_backward_grads(seqlen):
    q, k, v = _make(seqlen, requires_grad=True)
    dout = torch.randn_like(q)

    out = flash_attn_func(q, k, v)
    out.backward(dout)

    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    ref = _sdpa_fp32(qf, kf, vf)
    ref.backward(dout.float())

    for name, got, want in (
        ("dq", q.grad, qf.grad),
        ("dk", k.grad, kf.grad),
        ("dv", v.grad, vf.grad),
    ):
        d = (got.float() - want).abs().max().item()
        assert d < BWD_TOL, f"{name} maxdiff {d:.3e} at S={seqlen}"


@requires_cuda
def test_custom_softmax_scale():
    q, k, v = _make(256)
    scale = 0.05
    out = flash_attn_func(q, k, v, softmax_scale=scale)
    import torch.nn.functional as F

    ref = (
        F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
    )
    assert (out.float() - ref).abs().max().item() < FWD_TOL


@requires_cuda
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vs_fa4_when_available():
    """Cross-check against Tri Dao's flash_attn.cute when importable."""
    try:
        from flash_attn.cute import flash_attn_func as fa4_func
    except ImportError:
        pytest.skip("flash_attn.cute not importable")
    q, k, v = _make(1024)
    out, lse = flash_attn_func(q, k, v, return_lse=True)
    ref_out, ref_lse = fa4_func(q, k, v, return_lse=True)
    assert (out - ref_out).abs().max().item() < 5e-3
    assert (lse - ref_lse).abs().max().item() < 1e-5


@requires_cuda
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_causal_fwd(seqlen):
    q, k, v = _make(seqlen)
    with torch.no_grad():
        out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    import torch.nn.functional as F

    ref = (
        F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            is_causal=True,
        )
        .transpose(1, 2)
    )
    assert (out.float() - ref).abs().max().item() < 2e-2

    scale = q.shape[-1] ** -0.5
    scores = (
        torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * scale
    )
    tri = torch.ones(
        seqlen, seqlen, dtype=torch.bool, device="cuda"
    ).triu(1)
    ref_lse = torch.logsumexp(scores.masked_fill(tri, float("-inf")), dim=-1)
    assert (lse - ref_lse).abs().max().item() < LSE_TOL


@requires_cuda
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_causal_backward_grads(seqlen):
    q, k, v = _make(seqlen, requires_grad=True)
    dout = torch.randn_like(q)

    out = flash_attn_func(q, k, v, causal=True)
    out.backward(dout)

    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    import torch.nn.functional as F

    ref = (
        F.scaled_dot_product_attention(
            qf.transpose(1, 2), kf.transpose(1, 2), vf.transpose(1, 2),
            is_causal=True,
        )
        .transpose(1, 2)
    )
    ref.backward(dout.float())

    # Causal grads are noisier than non-causal (short rows average
    # fewer terms); FA4's own causal bwd differs from ours by only
    # ~4e-3 at the bench shape.
    for name, got, want in (
        ("dq", q.grad, qf.grad),
        ("dk", k.grad, kf.grad),
        ("dv", v.grad, vf.grad),
    ):
        d = (got.float() - want).abs().max().item()
        assert d < 5e-2, f"{name} maxdiff {d:.3e} at S={seqlen}"


@requires_cuda
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_causal_vs_fa4_when_available():
    try:
        from flash_attn.cute import flash_attn_func as fa4_func
    except ImportError:
        pytest.skip("flash_attn.cute not importable")
    q, k, v = _make(1024)
    with torch.no_grad():
        out, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    ref_out, ref_lse = fa4_func(q, k, v, causal=True, return_lse=True)
    assert (out - ref_out).abs().max().item() < 2e-2
    assert (lse - ref_lse).abs().max().item() < 1e-5


def _make_gqa(seqlen, hq=4, hkv=2, requires_grad=False):
    q = torch.randn(
        2, seqlen, hq, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=requires_grad,
    )
    k = torch.randn(
        2, seqlen, hkv, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=requires_grad,
    )
    v = torch.randn_like(k, requires_grad=requires_grad)
    return q, k, v


def _sdpa_gqa_fp32(q, k, v, causal=False):
    import torch.nn.functional as F

    g = q.shape[2] // k.shape[2]
    return (
        F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.repeat_interleave(g, dim=2).transpose(1, 2).float(),
            v.repeat_interleave(g, dim=2).transpose(1, 2).float(),
            is_causal=causal,
        )
        .transpose(1, 2)
    )


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_fwd(causal):
    q, k, v = _make_gqa(1024)
    out = flash_attn_func(q, k, v, causal=causal)
    ref = _sdpa_gqa_fp32(q, k, v, causal=causal)
    assert (out.float() - ref).abs().max().item() < 2e-2


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_backward_grads(causal):
    q, k, v = _make_gqa(640, requires_grad=True)
    dout = torch.randn_like(q)
    out = flash_attn_func(q, k, v, causal=causal)
    out.backward(dout)

    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    ref = _sdpa_gqa_fp32(qf, kf, vf, causal=causal)
    ref.backward(dout.float())
    for name, got, want in (
        ("dq", q.grad, qf.grad),
        ("dk", k.grad, kf.grad),
        ("dv", v.grad, vf.grad),
    ):
        d = (got.float() - want).abs().max().item()
        assert d < 5e-2, f"{name} maxdiff {d:.3e} causal={causal}"


@requires_cuda
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_gqa_vs_fa4_when_available():
    try:
        from flash_attn.cute import flash_attn_func as fa4_func
    except ImportError:
        pytest.skip("flash_attn.cute not importable")
    q, k, v = _make_gqa(1024)
    out = flash_attn_func(q, k, v)
    ref_out, _ = fa4_func(q, k, v, return_lse=True)
    assert (out - ref_out).abs().max().item() < 5e-3
