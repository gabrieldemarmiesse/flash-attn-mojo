"""Apple-GPU (Metal) forward-kernel tests.

Skipped off Apple silicon. The forward runs the fast 8x8
simdgroup-matrix Mojo kernel (`fwd_metal`); correctness is checked
against the fp32 pure-PyTorch reference. The Metal envelope is
non-causal, no window/softcap, seqlen % 128 == 0, head_dim in
{64, 128}, MHA or GQA; everything else routes to the reference.
"""

from __future__ import annotations

import pytest
import torch

from flash_attn_mojo import flash_attn_func, flash_attn_ref
from flash_attn_mojo.reference import flash_attn_varlen_ref  # noqa: F401

from conftest import requires_mps

# fp16 S-GEMM accumulate (kernel) vs fp32 reference — same band the
# CUDA fp16 path uses.
_TOL = 3e-3


def _qkv(B, S, Hq, Hkv, D):
    scale = 0.3
    q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="mps") * scale
    k = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="mps") * scale
    v = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="mps") * scale
    return q, k, v


@requires_mps
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("S", [128, 256, 512])
def test_metal_fwd_mha(S, D):
    q, k, v = _qkv(2, S, 8, 8, D)
    out, lse = flash_attn_func(q, k, v, return_lse=True)
    ref, rlse = flash_attn_ref(q, k, v, return_lse=True)
    assert (out.float() - ref.float()).abs().max().item() < _TOL
    assert (lse.float() - rlse.float()).abs().max().item() < _TOL


@requires_mps
@pytest.mark.parametrize("Hq,Hkv", [(8, 2), (16, 4), (6, 1)])
def test_metal_fwd_gqa(Hq, Hkv):
    q, k, v = _qkv(1, 384, Hq, Hkv, 128)
    out = flash_attn_func(q, k, v)
    ref = flash_attn_ref(q, k, v)
    assert (out.float() - ref.float()).abs().max().item() < _TOL


@requires_mps
def test_metal_fwd_matches_scale():
    q, k, v = _qkv(1, 256, 4, 4, 128)
    out = flash_attn_func(q, k, v, softmax_scale=0.05)
    ref = flash_attn_ref(q, k, v, softmax_scale=0.05)
    assert (out.float() - ref.float()).abs().max().item() < _TOL


@requires_mps
def test_metal_backward_matches_reference():
    """Forward is the Metal kernel; backward is the reference VJP —
    gradients must equal the pure-reference gradients exactly."""
    q, k, v = _qkv(2, 256, 8, 8, 128)

    def grads(fn):
        qq, kk, vv = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
        fn(qq, kk, vv).sum().backward()
        return qq.grad, kk.grad, vv.grad

    dq1, dk1, dv1 = grads(lambda a, b, c: flash_attn_func(a, b, c))
    dq2, dk2, dv2 = grads(lambda a, b, c: flash_attn_ref(a, b, c))
    assert torch.equal(dq1, dq2)
    assert torch.equal(dk1, dk2)
    assert torch.equal(dv1, dv2)


@requires_mps
def test_metal_causal_falls_back_to_reference():
    """No causal Metal kernel yet — the result must match the
    reference (the whole call routes through it)."""
    q, k, v = _qkv(1, 256, 4, 4, 128)
    out = flash_attn_func(q, k, v, causal=True)
    ref = flash_attn_ref(q, k, v, causal=True)
    assert torch.equal(out, ref)


@requires_mps
def test_metal_non_multiple_seqlen_falls_back():
    """head_dim=128 non-%128 seqlen routes to the reference."""
    q, k, v = _qkv(1, 200, 4, 4, 128)
    out = flash_attn_func(q, k, v)
    ref = flash_attn_ref(q, k, v)
    assert torch.equal(out, ref)
