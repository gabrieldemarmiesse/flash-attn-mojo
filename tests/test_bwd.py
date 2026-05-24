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


def test_backward_dropout_raises():
    B, L, H, D = 1, 32, 1, 64
    q, k, v = _make_qkv(B, L, H, D)
    out = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=0.1)
    dout = torch.randn_like(out)
    with pytest.raises(NotImplementedError, match="dropout"):
        out.backward(dout)
