"""Public-API correctness: `flash_attn_func` / `flash_attn_varlen_func`
end-to-end (autograd) against the pure-PyTorch references in
`flash_attn_mojo.reference`, across tail (S % 80 != 0) and exact-fit
seqlens, plus cross-checks vs Tri Dao's flash_attn.cute when
importable. bf16 noise-floor tolerances.

(Kernel-wrapper-level checks — what the bench delegates to — live in
test_kernels.py.)"""

from __future__ import annotations

import pytest
import torch

from flash_attn_mojo import flash_attn_func
from flash_attn_mojo.reference import flash_attn_ref, flash_attn_varlen_ref

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


def _ref(q, k, v, causal=False, return_lse=False, softmax_scale=None):
    """fp32 reference (handles GQA and LSE internally)."""
    return flash_attn_ref(
        q.float(), k.float(), v.float(),
        softmax_scale=softmax_scale, causal=causal, return_lse=return_lse,
    )


@requires_cuda
@pytest.mark.parametrize("seqlen", SEQLENS)
def test_fwd_out_and_lse(seqlen):
    q, k, v = _make(seqlen)
    out, lse = flash_attn_func(q, k, v, return_lse=True)
    ref, ref_lse = _ref(q, k, v, return_lse=True)
    assert (out.float() - ref).abs().max().item() < FWD_TOL
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
    ref = flash_attn_ref(qf, kf, vf)
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
    ref = _ref(q, k, v, softmax_scale=scale)
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
    ref, ref_lse = _ref(q, k, v, causal=True, return_lse=True)
    assert (out.float() - ref).abs().max().item() < 2e-2
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
    ref = flash_attn_ref(qf, kf, vf, causal=True)
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


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_gqa_fwd(causal):
    q, k, v = _make_gqa(1024)
    out = flash_attn_func(q, k, v, causal=causal)
    ref = _ref(q, k, v, causal=causal)
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
    ref = flash_attn_ref(qf, kf, vf, causal=causal)
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


# ---- varlen (packed cu_seqlens) ----
# Envelope: arbitrary lengths >= 1, self-attn lengths, MHA bwd.
VARLEN_SETS = [
    [128], [128, 256, 640], [1024, 128],
    [63, 129, 257], [128, 100],
]


def _make_varlen(lens, requires_grad=False):
    total = sum(lens)
    cu_list = [0]
    for L in lens:
        cu_list.append(cu_list[-1] + L)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    q = torch.randn(
        total, 4, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=requires_grad,
    )
    k = torch.randn_like(q, requires_grad=requires_grad)
    v = torch.randn_like(q, requires_grad=requires_grad)
    return q, k, v, cu


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens", VARLEN_SETS)
def test_varlen_fwd_out_and_lse(lens, causal):
    from flash_attn_mojo import flash_attn_varlen_func

    q, k, v, cu = _make_varlen(lens)
    with torch.no_grad():
        out, lse = flash_attn_varlen_func(
            q, k, v, cu, cu, causal=causal, return_lse=True
        )
    ref, ref_lse = flash_attn_varlen_ref(
        q.float(), k.float(), v.float(), cu, cu,
        causal=causal, return_lse=True,
    )
    assert (out.float() - ref).abs().max().item() < 2e-2
    assert (lse - ref_lse).abs().max().item() < LSE_TOL


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lens", VARLEN_SETS)
def test_varlen_backward_grads(lens, causal):
    from flash_attn_mojo import flash_attn_varlen_func

    q, k, v, cu = _make_varlen(lens, requires_grad=True)
    dout = torch.randn_like(q)
    out = flash_attn_varlen_func(q, k, v, cu, cu, causal=causal)
    out.backward(dout)

    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    ref = flash_attn_varlen_ref(qf, kf, vf, cu, cu, causal=causal)
    ref.backward(dout.float())
    for name, got, want in (
        ("dq", q.grad, qf.grad),
        ("dk", k.grad, kf.grad),
        ("dv", v.grad, vf.grad),
    ):
        d = (got.float() - want).abs().max().item()
        assert d < 5e-2, f"{name} maxdiff {d:.3e} lens={lens}"


@requires_cuda
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_vs_fa4_when_available(causal):
    try:
        from flash_attn.cute import flash_attn_varlen_func as fa4_varlen
    except ImportError:
        pytest.skip("flash_attn.cute not importable")
    from flash_attn_mojo import flash_attn_varlen_func

    q, k, v, cu = _make_varlen([1024, 256, 128])
    with torch.no_grad():
        out, lse = flash_attn_varlen_func(
            q, k, v, cu, cu, causal=causal, return_lse=True
        )
    ref_out, ref_lse = fa4_varlen(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=1024, max_seqlen_k=1024,
        causal=causal, return_lse=True,
    )
    assert (out - ref_out).abs().max().item() < 2e-2
    assert (lse - ref_lse).abs().max().item() < 1e-5


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_gqa_grads(causal):
    """Varlen GQA end-to-end autograd (ragged lengths included)."""
    from flash_attn_mojo import flash_attn_varlen_func

    lens = [256, 100, 384]
    cu_list = [0]
    for L in lens:
        cu_list.append(cu_list[-1] + L)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    total = cu_list[-1]
    q = torch.randn(
        total, 4, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        total, 2, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=True,
    )
    v = torch.randn_like(k, requires_grad=True)
    dout = torch.randn_like(q)

    out = flash_attn_varlen_func(q, k, v, cu, cu, causal=causal)
    out.backward(dout)

    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    ref = flash_attn_varlen_ref(qf, kf, vf, cu, cu, causal=causal)
    ref.backward(dout.float())
    assert (out.float() - ref).abs().max().item() < 2e-2
    for name, got, want in (
        ("dq", q.grad, qf.grad),
        ("dk", k.grad, kf.grad),
        ("dv", v.grad, vf.grad),
    ):
        d = (got.float() - want).abs().max().item()
        assert d < 5e-2, f"{name} maxdiff {d:.3e} causal={causal}"


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_hdim64_public_api(causal):
    """hdim64 through flash_attn_func end-to-end (autograd)."""
    q = torch.randn(
        2, 384, 4, 64, dtype=torch.bfloat16, device="cuda",
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    dout = torch.randn_like(q)
    out = flash_attn_func(q, k, v, causal=causal)
    out.backward(dout)
    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    ref = flash_attn_ref(qf, kf, vf, causal=causal)
    ref.backward(dout.float())
    assert (out.float() - ref).abs().max().item() < 2e-2
    for name, got, want in (
        ("dq", q.grad, qf.grad),
        ("dk", k.grad, kf.grad),
        ("dv", v.grad, vf.grad),
    ):
        d = (got.float() - want).abs().max().item()
        assert d < 5e-2, f"{name} maxdiff {d:.3e}"
