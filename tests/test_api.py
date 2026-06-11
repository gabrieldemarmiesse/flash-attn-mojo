"""Public-API tests: surface, envelope errors, CPU reference path."""

from __future__ import annotations

import pytest
import torch

import flash_attn_mojo
from flash_attn_mojo import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_ref,
)


def test_exports():
    assert set(flash_attn_mojo.__all__) == {
        "flash_attn_func",
        "flash_attn_kvpacked_func",
        "flash_attn_qkvpacked_func",
        "flash_attn_varlen_func",
        "flash_attn_ref",
    }


def _qkv(device="cpu", seqlen=128, head_dim=128, dtype=torch.bfloat16):
    q = torch.randn(2, seqlen, 4, head_dim, dtype=dtype, device=device)
    return q, torch.randn_like(q), torch.randn_like(q)


def test_causal_cpu_reference():
    q, k, v = _qkv()
    out = flash_attn_func(q, k, v, causal=True)
    ref = flash_attn_ref(q, k, v, causal=True)
    assert torch.equal(out, ref)


def test_envelope_errors():
    q, k, v = _qkv(dtype=torch.float32)
    with pytest.raises(ValueError, match="bf16 and fp16"):
        flash_attn_func(q, k, v)
    q, k, v = _qkv(head_dim=96)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attn_func(q, k, v)
    q, k, v = _qkv(seqlen=100)
    with pytest.raises(ValueError, match="multiple"):
        flash_attn_func(q, k, v)
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="MHA or GQA"):
        flash_attn_func(q, k[:, :, :2], v)


def test_cpu_reference_path():
    q, k, v = _qkv()
    out = flash_attn_func(q, k, v)
    ref = flash_attn_ref(q, k, v)
    assert torch.equal(out, ref)
    out2, lse = flash_attn_func(q, k, v, return_lse=True)
    assert torch.equal(out2, ref)
    assert lse.shape == (2, 4, 128) and lse.dtype == torch.float32


def test_packed_wrappers_cpu():
    q, k, v = _qkv()
    qkv = torch.stack([q, k, v], dim=2)
    ref = flash_attn_func(q, k, v)
    assert torch.equal(flash_attn_qkvpacked_func(qkv), ref)
    kv = torch.stack([k, v], dim=2)
    assert torch.equal(flash_attn_kvpacked_func(q, kv), ref)


def test_varlen_envelope_errors():
    from flash_attn_mojo import flash_attn_varlen_func

    q = torch.randn(256, 4, 128, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)

    with pytest.raises(ValueError, match="packed"):
        flash_attn_varlen_func(q.unsqueeze(0), k, v, cu, cu)
    with pytest.raises(ValueError, match="int32"):
        flash_attn_varlen_func(q, k, v, cu.long(), cu.long())
    with pytest.raises(ValueError, match="bf16"):
        flash_attn_varlen_func(q.float(), k.float(), v.float(), cu, cu)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attn_varlen_func(
            q[..., :64], k[..., :64], v[..., :64], cu, cu
        )


def test_varlen_cpu_reference_path():
    from flash_attn_mojo import flash_attn_varlen_func
    from flash_attn_mojo.reference import flash_attn_ref

    torch.manual_seed(0)
    q = torch.randn(384, 4, 128, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    cu = torch.tensor([0, 128, 384], dtype=torch.int32)
    out = flash_attn_varlen_func(q, k, v, cu, cu)
    ref0 = flash_attn_ref(
        q[:128].unsqueeze(0), k[:128].unsqueeze(0), v[:128].unsqueeze(0)
    )[0]
    assert (out[:128] - ref0).abs().max().item() < 1e-6


def test_varlen_value_envelope_errors():
    from flash_attn_mojo import flash_attn_varlen_func

    q = torch.randn(256, 4, 128, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)

    with pytest.raises(ValueError, match="self-attention only"):
        flash_attn_varlen_func(
            q, k, v, cu, torch.tensor([0, 64, 256], dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="zero-length"):
        flash_attn_varlen_func(
            q, k, v,
            torch.tensor([0, 0, 256], dtype=torch.int32),
            torch.tensor([0, 0, 256], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="start at 0"):
        flash_attn_varlen_func(
            q, k, v,
            torch.tensor([128, 256], dtype=torch.int32),
            torch.tensor([128, 256], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="total_tokens"):
        flash_attn_varlen_func(
            q, k, v,
            torch.tensor([0, 128], dtype=torch.int32),
            torch.tensor([0, 128], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="max_seqlen_q"):
        flash_attn_varlen_func(q, k, v, cu, cu, max_seqlen_q=64)
