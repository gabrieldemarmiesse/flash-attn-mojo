"""Forward-kernel correctness tests.

bf16 + head_dim=64 envelope. Cross-checks against PyTorch's SDPA (fp32
reference) and, when available, upstream Tri Dao flash-attn.
"""

from __future__ import annotations

import pytest
import torch

import flash_attn_mojo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fwd kernel needs cuda"
)


def _sdpa_ref(q, k, v, causal: bool):
    """Pure-PyTorch SDPA reference in fp32, returned in q's dtype."""
    q_h = q.transpose(1, 2).float()
    k_h = k.transpose(1, 2).float()
    v_h = v.transpose(1, 2).float()
    out = torch.nn.functional.scaled_dot_product_attention(
        q_h, k_h, v_h, is_causal=causal
    )
    return out.transpose(1, 2).contiguous().to(q.dtype)


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("L", [16, 32, 64, 128, 512])
def test_causal_correctness_vs_sdpa(L):
    """mojo causal vs fp32 SDPA causal reference."""
    # H=1 keeps the (L, D) inner block contiguous for any L: the
    # launcher's pad-and-copy path requires q.stride(1) == head_dim,
    # which (B, L, H, D) contig only satisfies when H == 1.
    B, H, D = 2, 1, 64
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    out_ref = _sdpa_ref(q, k, v, causal=True)

    diff = _max_abs(out_mojo, out_ref)
    # bf16 tol, similar to upstream's own bf16 flash-attn diff vs fp32.
    assert diff < 5e-2, f"L={L} causal vs SDPA: max |diff|={diff:.3e}"


@pytest.mark.parametrize("L", [16, 32, 64, 128, 512])
def test_causal_matches_upstream_flash_attn(L):
    """mojo causal must agree with upstream flash-attn 2 causal within
    ~1.5x of upstream's own error vs fp32 SDPA."""
    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        pytest.skip("upstream flash-attn not available")

    # H=1 keeps the (L, D) inner block contiguous for any L: the
    # launcher's pad-and-copy path requires q.stride(1) == head_dim,
    # which (B, L, H, D) contig only satisfies when H == 1.
    B, H, D = 2, 1, 64
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(q, k, v, causal=True)
    out_up = upstream_fwd(q, k, v, causal=True)
    out_ref = _sdpa_ref(q, k, v, causal=True)

    diff_mojo = _max_abs(out_mojo, out_ref)
    diff_up = _max_abs(out_up, out_ref)
    # Allow up to 1.5x upstream's own error vs fp32, plus a small floor
    # for very-small-L runs where upstream's diff is already < 1e-3.
    tol = max(1.5 * diff_up, 5e-3)
    assert diff_mojo < tol, (
        f"L={L} causal mojo-vs-SDPA={diff_mojo:.3e} "
        f"vs upstream-vs-SDPA={diff_up:.3e}, tol={tol:.3e}"
    )


@pytest.mark.parametrize("nheads_q,nheads_kv", [(8, 1), (8, 2), (8, 4), (8, 8)])
@pytest.mark.parametrize("L", [128, 512])
def test_mqa_gqa_correctness(nheads_q, nheads_kv, L):
    """MQA/GQA: mojo with nheads_kv < nheads_q must match SDPA-with-
    repeat_interleave and (when available) upstream flash-attn."""
    B, D = 2, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, nheads_q, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, nheads_kv, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, nheads_kv, D, dtype=torch.bfloat16, device="cuda")

    # Pure-PyTorch fp32 ref with k/v repeat-interleaved to nheads_q.
    group = nheads_q // nheads_kv
    k_rep = k.repeat_interleave(group, dim=2)
    v_rep = v.repeat_interleave(group, dim=2)
    out_ref = _sdpa_ref(q, k_rep, v_rep, causal=False)

    out_mojo = flash_attn_mojo.flash_attn_func(q, k, v, causal=False)
    diff = _max_abs(out_mojo, out_ref)
    assert diff < 5e-2, (
        f"MQA/GQA mojo vs SDPA: nheads=({nheads_q},{nheads_kv}) L={L} "
        f"max |diff|={diff:.3e}"
    )

    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        return
    out_up = upstream_fwd(q, k, v, causal=False)
    diff_up = _max_abs(out_mojo, out_up)
    assert diff_up < 5e-2, (
        f"MQA/GQA mojo vs upstream: nheads=({nheads_q},{nheads_kv}) L={L} "
        f"max |diff|={diff_up:.3e}"
    )


@pytest.mark.parametrize("softcap", [10.0, 30.0, 50.0])
@pytest.mark.parametrize("L", [128, 512])
@pytest.mark.parametrize("causal", [False, True])
def test_softcap_correctness(softcap, L, causal):
    """Softcap: mojo must agree with upstream flash-attn (the only ref
    that supports softcap directly)."""
    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        pytest.skip("upstream flash-attn not available")
    B, H, D = 2, 1, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=causal, softcap=softcap
    )
    out_up = upstream_fwd(q, k, v, causal=causal, softcap=softcap)
    diff = _max_abs(out_mojo, out_up)
    assert diff < 5e-2, (
        f"softcap mojo vs upstream: softcap={softcap} L={L} causal={causal} "
        f"max |diff|={diff:.3e}"
    )


def test_softcap_zero_unchanged():
    """softcap=0 must produce the exact same output as the default
    (regression: don't perturb the no-softcap fast path)."""
    B, L, H, D = 2, 128, 1, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_default = flash_attn_mojo.flash_attn_func(q, k, v, causal=False)
    out_zero = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=False, softcap=0.0
    )
    assert torch.equal(out_default, out_zero)


@pytest.mark.parametrize("causal", [False, True])
def test_return_attn_probs(causal):
    """`return_attn_probs=True` must yield an LSE that matches both
    upstream flash-attn's `softmax_lse` and a pure-fp32 reference."""
    B, L, H, D = 2, 128, 2, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo, lse_mojo, rng = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=causal, return_attn_probs=True
    )
    assert rng is None
    assert lse_mojo.shape == (B, H, L)
    assert lse_mojo.dtype == torch.float32
    assert torch.isfinite(lse_mojo).all(), "LSE has non-finite entries"

    # Pure-fp32 reference: lse[b, h, q] = log(sum_j exp(scale * s_ij))
    # over valid (non-masked) keys.
    scale = D ** -0.5
    q_f = q.transpose(1, 2).float()  # (B, H, L, D)
    k_f = k.transpose(1, 2).float()
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # (B, H, L, L)
    if causal:
        mask = torch.triu(
            torch.ones(L, L, device="cuda", dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1)  # (B, H, L)

    diff_ref = (lse_mojo - lse_ref).abs().max().item()
    assert diff_ref < 1e-2, (
        f"causal={causal} mojo LSE vs fp32 ref: max|diff|={diff_ref:.3e}"
    )

    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        return
    out_up, lse_up, _ = upstream_fwd(q, k, v, causal=causal, return_attn_probs=True)
    diff_up = (lse_mojo - lse_up).abs().max().item()
    assert diff_up < 1e-2, (
        f"causal={causal} mojo LSE vs upstream: max|diff|={diff_up:.3e}"
    )


@pytest.mark.parametrize("L", [16, 32, 64, 128, 512])
def test_noncausal_regression(L):
    """Non-causal path must still match SDPA (regression for the
    causal plumbing not touching causal=False)."""
    # H=1 keeps the (L, D) inner block contiguous for any L: the
    # launcher's pad-and-copy path requires q.stride(1) == head_dim,
    # which (B, L, H, D) contig only satisfies when H == 1.
    B, H, D = 2, 1, 64
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(q, k, v, causal=False)
    out_ref = _sdpa_ref(q, k, v, causal=False)
    diff = _max_abs(out_mojo, out_ref)
    assert diff < 5e-2, f"L={L} non-causal vs SDPA: max |diff|={diff:.3e}"
