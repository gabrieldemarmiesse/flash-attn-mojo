"""Kernel-level correctness — the bench's check suites.

Every JIT kernel variant (fwd/bwd x dense/varlen x plain/causal x
mha/gqa) against the pure-PyTorch fp32 reference
(`flash_attn_mojo.reference`), plus canonical-bench-shape
cross-checks against Tri Dao's `flash_attn.cute` when importable.

`scripts/bench_fa4.py --check / --check-only` delegates here with a
``-k`` expression built from its flags (e.g. ``bwd and dense and
causal and mha``) — test/parameter names are chosen so those four
axes select cleanly. Set ``FLASH_ATTN_MOJO_TEST_IMPL=fa4`` to run
the same reference checks against Tri Dao's kernels instead
(harness validation; canonical cross-checks are skipped — they ARE
the mojo-vs-fa4 comparison).
"""

from __future__ import annotations

import os

import pytest
import torch

from flash_attn_mojo.reference import flash_attn_ref, flash_attn_varlen_ref

from conftest import requires_cuda

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

IMPL = os.environ.get("FLASH_ATTN_MOJO_TEST_IMPL", "mojo")

# 640 = 8*80 exercises the bwd tile_m=80 exact-fit path; the others
# leave partial tail m-tiles. 128 is the minimum supported seqlen.
SEQLENS = [128, 256, 640, 1024]
# Tile-aligned varlen check sets (every length % 128 == 0).
VARLEN_SETS = [[128], [128, 256, 640], [1024, 128], [256] * 16]
VARLEN_IDS = ["L128", "L128-256-640", "L1024-128", "L256x16"]
# Canonical bench shapes (B, S, H, D) and the canonical mixed
# varlen config (8 seqs, 16384 tokens).
CANONICAL = (2, 8192, 16, 128)
CANONICAL_HKV = 4
CANONICAL_LENS = [3072, 2816, 2560, 2048, 1792, 1536, 1280, 1280]

FWD_TOL = 5e-3
FWD_TOL_MASKED = 2e-2  # causal/GQA rows average fewer terms
LSE_TOL = 1e-5
BWD_TOL = 2e-2
BWD_TOL_MASKED = 5e-2


def _skip_if_impl_unavailable():
    if IMPL == "fa4":
        pytest.importorskip("flash_attn.cute")


def _fwd(q, k, v, causal):
    """Forward of the impl under test -> (out, lse)."""
    if IMPL == "fa4":
        from flash_attn.cute import flash_attn_func

        return flash_attn_func(q, k, v, causal=causal, return_lse=True)
    from flash_attn_mojo.fwd_fa4 import fa4_fwd

    return fa4_fwd(q, k, v, causal=causal)


def _bwd(q, k, v, out, dout, lse, causal):
    if IMPL == "fa4":
        from flash_attn.cute.interface import _flash_attn_bwd

        return _flash_attn_bwd(q, k, v, out, dout, lse, causal=causal)
    from flash_attn_mojo.bwd_fa4 import bwd_fa4

    return bwd_fa4(q, k, v, out, dout, lse, causal=causal)


def _varlen_fwd(q, k, v, cu, causal):
    if IMPL == "fa4":
        from flash_attn.cute import flash_attn_varlen_func

        max_len = int((cu[1:] - cu[:-1]).max())
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=causal, return_lse=True,
        )
    from flash_attn_mojo.fwd_fa4 import fa4_varlen_fwd

    return fa4_varlen_fwd(q, k, v, cu, cu, causal=causal)


def _varlen_bwd(q, k, v, out, dout, lse, cu, causal):
    if IMPL == "fa4":
        from flash_attn.cute.interface import _flash_attn_bwd

        max_len = int((cu[1:] - cu[:-1]).max())
        return _flash_attn_bwd(
            q, k, v, out, dout, lse,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=causal,
        )
    from flash_attn_mojo.bwd_fa4 import bwd_fa4_varlen

    return bwd_fa4_varlen(q, k, v, out, dout, lse, cu, cu, causal=causal)


def _make(seqlen, hq=4, hkv=4, batch=2, requires_grad=False):
    q = torch.randn(
        batch, seqlen, hq, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=requires_grad,
    )
    k = torch.randn(
        batch, seqlen, hkv, 128, dtype=torch.bfloat16, device="cuda",
        requires_grad=requires_grad,
    )
    v = torch.randn_like(k, requires_grad=requires_grad)
    return q, k, v


def _make_varlen(lens, h=4):
    cu_list = [0]
    for L in lens:
        cu_list.append(cu_list[-1] + L)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    q = torch.randn(
        cu_list[-1], h, 128, dtype=torch.bfloat16, device="cuda"
    )
    return q, torch.randn_like(q), torch.randn_like(q), cu


# ---------------------------------------------------------- dense
@requires_cuda
@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("heads", ["mha", "gqa"])
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_fwd_dense(seqlen, mask, heads):
    _skip_if_impl_unavailable()
    torch.manual_seed(1)
    causal = mask == "causal"
    q, k, v = _make(seqlen, hkv=4 if heads == "mha" else 2)
    out, lse = _fwd(q, k, v, causal)
    ref, ref_lse = flash_attn_ref(
        q.float(), k.float(), v.float(), causal=causal, return_lse=True
    )
    tol = FWD_TOL if (mask, heads) == ("plain", "mha") else FWD_TOL_MASKED
    d = (out.float() - ref).abs().max().item()
    assert d < tol, f"out maxdiff {d:.3e}"
    dl = (lse - ref_lse).abs().max().item()
    assert dl < LSE_TOL, f"lse maxdiff {dl:.3e}"


@requires_cuda
@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("heads", ["mha", "gqa"])
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_bwd_dense(seqlen, mask, heads):
    _skip_if_impl_unavailable()
    torch.manual_seed(1)
    causal = mask == "causal"
    q, k, v = _make(seqlen, hkv=4 if heads == "mha" else 2)
    dout = torch.randn_like(q)
    out, lse = _fwd(q, k, v, causal)
    dq, dk, dv = _bwd(q, k, v, out, dout, lse, causal)

    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    ref = flash_attn_ref(qf, kf, vf, causal=causal)
    ref.backward(dout.float())
    tol = BWD_TOL if (mask, heads) == ("plain", "mha") else BWD_TOL_MASKED
    for name, got, want in (
        ("dq", dq, qf.grad), ("dk", dk, kf.grad), ("dv", dv, vf.grad)
    ):
        d = (got.float() - want).abs().max().item()
        assert d < tol, f"{name} maxdiff {d:.3e} at S={seqlen}"


# --------------------------------------------------------- varlen
@requires_cuda
@pytest.mark.parametrize("lens", VARLEN_SETS, ids=VARLEN_IDS)
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_fwd_varlen(lens, mask):
    _skip_if_impl_unavailable()
    torch.manual_seed(3)
    causal = mask == "causal"
    q, k, v, cu = _make_varlen(lens)
    out, lse = _varlen_fwd(q, k, v, cu, causal)
    ref, ref_lse = flash_attn_varlen_ref(
        q.float(), k.float(), v.float(), cu, cu,
        causal=causal, return_lse=True,
    )
    tol = FWD_TOL if mask == "plain" else FWD_TOL_MASKED
    d = (out.float() - ref).abs().max().item()
    assert d < tol, f"out maxdiff {d:.3e}"
    dl = (lse - ref_lse).abs().max().item()
    assert dl < LSE_TOL, f"lse maxdiff {dl:.3e}"


@requires_cuda
@pytest.mark.parametrize("lens", VARLEN_SETS, ids=VARLEN_IDS)
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_bwd_varlen(lens, mask):
    _skip_if_impl_unavailable()
    torch.manual_seed(3)
    causal = mask == "causal"
    q, k, v, cu = _make_varlen(lens)
    dout = torch.randn_like(q)
    out, lse = _varlen_fwd(q, k, v, cu, causal)
    dq, dk, dv = _varlen_bwd(q, k, v, out, dout, lse, cu, causal)

    qf = q.detach().float().requires_grad_()
    kf = k.detach().float().requires_grad_()
    vf = v.detach().float().requires_grad_()
    ref = flash_attn_varlen_ref(qf, kf, vf, cu, cu, causal=causal)
    ref.backward(dout.float())
    tol = BWD_TOL if mask == "plain" else BWD_TOL_MASKED
    for name, got, want in (
        ("dq", dq, qf.grad), ("dk", dk, kf.grad), ("dv", dv, vf.grad)
    ):
        d = (got.float() - want).abs().max().item()
        assert d < tol, f"{name} maxdiff {d:.3e} lens={lens}"


# ----------------------- canonical-shape cross-checks (mojo vs fa4)
def _skip_unless_cross_check():
    if IMPL != "mojo":
        pytest.skip("canonical cross-check IS the mojo-vs-fa4 compare")
    pytest.importorskip("flash_attn.cute")


@requires_cuda
@pytest.mark.parametrize("heads", ["mha", "gqa"])
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_fwd_dense_canonical(mask, heads):
    """Mojo vs FA4 at the exact benchmark shape (what the old
    bench-shape --check compared)."""
    _skip_unless_cross_check()
    from flash_attn.cute import flash_attn_func as fa4_func
    from flash_attn_mojo.fwd_fa4 import fa4_fwd

    torch.manual_seed(0)
    causal = mask == "causal"
    B, S, H, D = CANONICAL
    q, k, v = _make(S, hq=H, hkv=H if heads == "mha" else CANONICAL_HKV,
                    batch=B)
    out, lse = fa4_fwd(q, k, v, causal=causal)
    ref_out, ref_lse = fa4_func(q, k, v, causal=causal, return_lse=True)
    assert (out - ref_out).abs().max().item() < FWD_TOL_MASKED
    assert (lse - ref_lse).abs().max().item() < LSE_TOL


@requires_cuda
@pytest.mark.parametrize("heads", ["mha", "gqa"])
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_bwd_dense_canonical(mask, heads):
    _skip_unless_cross_check()
    from flash_attn.cute import flash_attn_func as fa4_func
    from flash_attn.cute.interface import _flash_attn_bwd
    from flash_attn_mojo.bwd_fa4 import bwd_fa4

    torch.manual_seed(0)
    causal = mask == "causal"
    B, S, H, D = CANONICAL
    q, k, v = _make(S, hq=H, hkv=H if heads == "mha" else CANONICAL_HKV,
                    batch=B)
    dout = torch.randn_like(q)
    # Identical out/lse inputs for both backwards (FA4's fwd).
    out, lse = fa4_func(q, k, v, causal=causal, return_lse=True)
    grads = bwd_fa4(q, k, v, out, dout, lse, causal=causal)
    refs = _flash_attn_bwd(q, k, v, out, dout, lse, causal=causal)
    for name, got, ref in zip(("dq", "dk", "dv"), grads, refs):
        d = (got - ref).abs().max().item()
        assert d < BWD_TOL_MASKED, f"{name} maxdiff {d:.3e}"


@requires_cuda
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_fwd_varlen_canonical(mask):
    _skip_unless_cross_check()
    from flash_attn.cute import flash_attn_varlen_func as fa4_varlen
    from flash_attn_mojo.fwd_fa4 import fa4_varlen_fwd

    torch.manual_seed(0)
    causal = mask == "causal"
    q, k, v, cu = _make_varlen(CANONICAL_LENS, h=16)
    out, lse = fa4_varlen_fwd(q, k, v, cu, cu, causal=causal)
    max_len = max(CANONICAL_LENS)
    ref_out, ref_lse = fa4_varlen(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_len, max_seqlen_k=max_len,
        causal=causal, return_lse=True,
    )
    assert (out - ref_out).abs().max().item() < FWD_TOL_MASKED
    assert (lse - ref_lse).abs().max().item() < LSE_TOL


@requires_cuda
@pytest.mark.parametrize("mask", ["plain", "causal"])
def test_bwd_varlen_canonical(mask):
    _skip_unless_cross_check()
    from flash_attn.cute import flash_attn_varlen_func as fa4_varlen
    from flash_attn.cute.interface import _flash_attn_bwd
    from flash_attn_mojo.bwd_fa4 import bwd_fa4_varlen

    torch.manual_seed(0)
    causal = mask == "causal"
    q, k, v, cu = _make_varlen(CANONICAL_LENS, h=16)
    dout = torch.randn_like(q)
    max_len = max(CANONICAL_LENS)
    out, lse = fa4_varlen(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_len, max_seqlen_k=max_len,
        causal=causal, return_lse=True,
    )
    grads = bwd_fa4_varlen(q, k, v, out, dout, lse, cu, cu, causal=causal)
    refs = _flash_attn_bwd(
        q, k, v, out, dout, lse,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_len, max_seqlen_k=max_len,
        causal=causal,
    )
    for name, got, ref in zip(("dq", "dk", "dv"), grads, refs):
        d = (got - ref).abs().max().item()
        assert d < BWD_TOL_MASKED, f"{name} maxdiff {d:.3e}"
