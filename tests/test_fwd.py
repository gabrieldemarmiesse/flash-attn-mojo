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
