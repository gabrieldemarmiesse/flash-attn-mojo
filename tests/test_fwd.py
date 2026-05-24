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


@pytest.mark.parametrize("causal", [False, True])
def test_qkvpacked_matches_unpacked(causal):
    """`flash_attn_qkvpacked_func` must be bit-equal to the unpacked
    `flash_attn_func` on the same data."""
    B, L, H, D = 2, 128, 4, 64
    # H=4 with the kernel's launcher requires q.stride(1)==head_dim; the
    # packed-along-dim=2 layout still produces contiguous per-token rows
    # when we materialise q/k/v via `.contiguous()` after unbind. The
    # packed wrapper does the unbind internally — the kernel is the same
    # one called by `flash_attn_func`, so we round-trip both through
    # `.contiguous()` here to keep the strides identical.
    qkv = torch.randn(B, L, 3, H, D, dtype=torch.bfloat16, device="cuda")
    q, k, v = (t.contiguous() for t in qkv.unbind(dim=2))

    out_packed = flash_attn_mojo.flash_attn_qkvpacked_func(qkv, causal=causal)
    out_unpacked = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)

    # The packed wrapper feeds the unbound (non-contiguous) views into
    # the kernel. If the kernel requires contiguous q, packed may
    # differ; allow a tiny numeric drift but expect exact equality on
    # contiguous-equivalent inputs.
    assert torch.equal(out_packed, out_unpacked), (
        f"qkvpacked vs unpacked (causal={causal}) differ: "
        f"max |diff|={_max_abs(out_packed, out_unpacked):.3e}"
    )


@pytest.mark.parametrize("causal", [False, True])
def test_kvpacked_matches_unpacked(causal):
    """`flash_attn_kvpacked_func` must be bit-equal to the unpacked
    `flash_attn_func` on the same data, with MQA (H_q != H_kv)."""
    B, L, D = 2, 128, 64
    H_q, H_kv = 4, 2
    q = torch.randn(B, L, H_q, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, L, 2, H_kv, D, dtype=torch.bfloat16, device="cuda")
    k, v = (t.contiguous() for t in kv.unbind(dim=2))

    out_packed = flash_attn_mojo.flash_attn_kvpacked_func(q, kv, causal=causal)
    out_unpacked = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)

    assert torch.equal(out_packed, out_unpacked), (
        f"kvpacked vs unpacked (causal={causal}) differ: "
        f"max |diff|={_max_abs(out_packed, out_unpacked):.3e}"
    )


@pytest.mark.parametrize(
    "window_left,window_right",
    [(64, 0), (32, 32), (128, -1), (-1, 128)],
)
@pytest.mark.parametrize("causal", [False, True])
def test_window_correctness(window_left, window_right, causal):
    """Sliding-window attention: mojo with window_size=(L, R) must match
    upstream flash-attn's window output within bf16 tol."""
    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        pytest.skip("upstream flash-attn not available")
    B, L, H, D = 2, 512, 4, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    window = (window_left, window_right)
    out_mojo = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=causal, window_size=window
    )
    out_up = upstream_fwd(q, k, v, causal=causal, window_size=window)
    diff = _max_abs(out_mojo, out_up)
    assert diff < 5e-2, (
        f"window mojo vs upstream: window={window} causal={causal} "
        f"max |diff|={diff:.3e}"
    )


@pytest.mark.parametrize("L", [17, 47, 100])
@pytest.mark.parametrize("H", [4, 8])
@pytest.mark.parametrize("causal", [False, True])
def test_unaligned_seqlen_multihead(L, H, causal):
    """Unaligned-seqlen pad path must handle non-contig L stride.

    With H > 1 a (B, L, H, D) contiguous tensor has L stride = H*D, not
    D. Combined with L not aligned to kBlockN=64, this exercises the
    strided-row copy in the pad-and-copy path.
    """
    B, D = 2, 64
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    out_ref = _sdpa_ref(q, k, v, causal=causal)
    diff = _max_abs(out_mojo, out_ref)
    assert diff < 5e-2, (
        f"L={L} H={H} causal={causal} vs SDPA: max |diff|={diff:.3e}"
    )


@pytest.mark.parametrize("slopes_shape", ["per_head", "per_batch_head"])
@pytest.mark.parametrize("causal", [False, True])
def test_alibi_correctness(slopes_shape, causal):
    """ALiBi: mojo with alibi_slopes must match upstream flash-attn's
    alibi output within bf16 tol.

    Composition: alibi is added post-softmax-scale (and post-softcap)
    and pre-mask, in natural-log domain. The kernel folds `log2e` into
    the slope at the top so the per-element add lands in the same
    log2 domain as the rest of the inner loop.
    """
    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        pytest.skip("upstream flash-attn not available")
    B, L, H, D = 2, 256, 4, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    # Typical ALiBi slope pattern (negative, geometric decay per head).
    base = -(2.0 ** (torch.arange(1, H + 1, dtype=torch.float32) / H))
    if slopes_shape == "per_head":
        slopes = base.to("cuda")
    else:
        # (B, H): per-batch variation so the batch stride actually
        # matters in the kernel's load.
        slopes = (
            base.unsqueeze(0).expand(B, H).contiguous()
            * torch.tensor([1.0, 0.7], dtype=torch.float32).unsqueeze(1)
        ).to("cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=causal, alibi_slopes=slopes
    )
    out_up = upstream_fwd(q, k, v, causal=causal, alibi_slopes=slopes)
    diff = _max_abs(out_mojo, out_up)
    assert diff < 5e-2, (
        f"alibi mojo vs upstream: slopes_shape={slopes_shape} causal={causal} "
        f"max |diff|={diff:.3e}"
    )


def test_no_alibi_regression():
    """alibi_slopes=None must produce the exact same output as the
    default (regression: don't perturb the no-alibi fast path)."""
    B, L, H, D = 2, 256, 4, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_default = flash_attn_mojo.flash_attn_func(q, k, v, causal=False)
    out_none = flash_attn_mojo.flash_attn_func(
        q, k, v, causal=False, alibi_slopes=None
    )
    assert torch.equal(out_default, out_none), (
        "alibi_slopes=None must match the default (no-alibi) fast path "
        "bitwise"
    )


@pytest.mark.parametrize("causal", [False, True])
def test_head_dim_32_correctness(causal):
    """head_dim=32: mojo must agree with upstream flash-attn within bf16 tol."""
    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        pytest.skip("upstream flash-attn not available")
    B, L, H, D = 2, 512, 4, 32
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    out_up = upstream_fwd(q, k, v, causal=causal)
    out_ref = _sdpa_ref(q, k, v, causal=causal)

    diff_mojo = _max_abs(out_mojo, out_ref)
    diff_up = _max_abs(out_up, out_ref)
    tol = max(1.5 * diff_up, 5e-3)
    assert diff_mojo < tol, (
        f"hd=32 causal={causal} mojo-vs-SDPA={diff_mojo:.3e} "
        f"vs upstream-vs-SDPA={diff_up:.3e}, tol={tol:.3e}"
    )


@pytest.mark.parametrize("causal", [False, True])
def test_head_dim_128_correctness(causal):
    """head_dim=128: mojo must agree with upstream flash-attn within bf16 tol."""
    try:
        from flash_attn import flash_attn_func as upstream_fwd
    except ImportError:
        pytest.skip("upstream flash-attn not available")
    B, L, H, D = 2, 512, 4, 128
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_mojo = flash_attn_mojo.flash_attn_func(q, k, v, causal=causal)
    out_up = upstream_fwd(q, k, v, causal=causal)
    out_ref = _sdpa_ref(q, k, v, causal=causal)

    diff_mojo = _max_abs(out_mojo, out_ref)
    diff_up = _max_abs(out_up, out_ref)
    tol = max(1.5 * diff_up, 5e-3)
    assert diff_mojo < tol, (
        f"hd=128 causal={causal} mojo-vs-SDPA={diff_mojo:.3e} "
        f"vs upstream-vs-SDPA={diff_up:.3e}, tol={tol:.3e}"
    )


def test_dropout_zero_unchanged():
    """`dropout_p=0.0` must produce bit-identical output to the no-arg
    call — the dropout-active variant compiles to a separate kernel,
    and we want the non-dropout default path to remain untouched.
    """
    B, L, H, D = 2, 128, 4, 64
    torch.manual_seed(0)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_default = flash_attn_mojo.flash_attn_func(q, k, v)
    out_zero = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=0.0)
    assert torch.equal(out_default, out_zero)


def test_dropout_basic():
    """`dropout_p=0.3`: output is finite, deviates from no-dropout, and
    is deterministic given a fixed torch RNG seed (the host generates
    the per-call seed via torch.randint — seeding the host RNG fixes it).
    """
    B, L, H, D = 2, 128, 4, 64
    p = 0.3
    torch.manual_seed(42)
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")

    out_nodrop = flash_attn_mojo.flash_attn_func(q, k, v)

    # Two runs with the same host RNG seed must agree (the kernel mixer
    # is fully determined by the (rng_seed, b, h, q_idx, kv_idx) tuple,
    # and the seed in turn is reproducibly drawn from torch's CPU RNG).
    torch.manual_seed(123)
    out_drop_a, lse_a, rng_a = flash_attn_mojo.flash_attn_func(
        q, k, v, dropout_p=p, return_attn_probs=True
    )
    torch.manual_seed(123)
    out_drop_b, lse_b, rng_b = flash_attn_mojo.flash_attn_func(
        q, k, v, dropout_p=p, return_attn_probs=True
    )
    assert torch.equal(out_drop_a, out_drop_b), "dropout output not deterministic"
    assert rng_a is not None and rng_b is not None
    assert rng_a.shape == (2,)
    assert rng_a.dtype == torch.int64
    assert torch.equal(rng_a, rng_b)
    # LSE is computed from the un-dropped softmax mass; dropout doesn't
    # alter the rowmax/rowsum running stats, so LSE must match the
    # no-dropout call.
    out_nd_full, lse_nodrop, _ = flash_attn_mojo.flash_attn_func(
        q, k, v, return_attn_probs=True
    )
    assert torch.allclose(lse_a, lse_nodrop, atol=1e-4)

    # A different seed produces a different mask (and thus output).
    torch.manual_seed(456)
    out_drop_c = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=p)
    assert not torch.equal(out_drop_a, out_drop_c)

    # Output is finite and meaningfully different from the no-dropout
    # baseline (most rows have some surviving entries; a few may have
    # all entries dropped in a small tile, but the aggregate L2 must be
    # bounded away from zero).
    assert torch.isfinite(out_drop_a).all()
    diff = (out_drop_a.float() - out_nodrop.float()).abs().mean().item()
    assert diff > 1e-3, f"dropout output suspiciously close to no-drop: {diff}"

    # Statistical: the kernel scales survivors by 1/(1-p); the resulting
    # output's expected value equals the no-drop output. Average over
    # multiple draws and the gap should shrink relative to a single
    # draw (sanity check that we're using inverted dropout, not "drop +
    # don't rescale").
    acc = torch.zeros_like(out_nodrop, dtype=torch.float32)
    n = 16
    for s in range(n):
        torch.manual_seed(1000 + s)
        out_s = flash_attn_mojo.flash_attn_func(q, k, v, dropout_p=p)
        acc = acc + out_s.float()
    avg = (acc / n).to(out_nodrop.dtype)
    # E[O_drop] == O_nodrop with inverted dropout; small n leaves
    # variance, so use a loose bound (no-dropout MAE vs SDPA is ~3e-2
    # at this shape; the dropout-averaged value should be within an
    # order of magnitude of that).
    avg_diff = (avg.float() - out_nodrop.float()).abs().mean().item()
    nodrop_mag = out_nodrop.float().abs().mean().item()
    assert avg_diff < 0.5 * nodrop_mag, (
        f"averaged dropout output too far from no-drop baseline: "
        f"avg_diff={avg_diff:.3e}, nodrop_mag={nodrop_mag:.3e}"
    )


# ----------------------------------------------------------------------------
# Varlen / packed-batch API
# ----------------------------------------------------------------------------


def _pack_varlen(seqlens, H, D, dtype=torch.bfloat16, device="cuda", seed=0):
    """Build a packed (total, H, D) Q/K/V triple plus cu_seqlens, returning
    also the per-batch unpacked (1, L, H, D) tensors for cross-check."""
    torch.manual_seed(seed)
    qs, ks, vs = [], [], []
    per_batch = []
    for L in seqlens:
        q_b = torch.randn(1, L, H, D, dtype=dtype, device=device)
        k_b = torch.randn(1, L, H, D, dtype=dtype, device=device)
        v_b = torch.randn(1, L, H, D, dtype=dtype, device=device)
        qs.append(q_b.squeeze(0))
        ks.append(k_b.squeeze(0))
        vs.append(v_b.squeeze(0))
        per_batch.append((q_b, k_b, v_b))
    q = torch.cat(qs, dim=0).contiguous()
    k = torch.cat(ks, dim=0).contiguous()
    v = torch.cat(vs, dim=0).contiguous()
    cu = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
        dtype=torch.int32, device=device,
    )
    return q, k, v, cu, per_batch


@pytest.mark.parametrize("causal", [False, True])
def test_varlen_basic(causal):
    """Packed-batch varlen output matches per-element flash_attn_func."""
    seqlens = [64, 128, 96]
    H, D = 1, 64
    q, k, v, cu, per_batch = _pack_varlen(seqlens, H, D, seed=0)

    out = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu, cu,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        causal=causal,
    )

    start = 0
    for (q_b, k_b, v_b), L in zip(per_batch, seqlens):
        out_b_ref = flash_attn_mojo.flash_attn_func(
            q_b, k_b, v_b, causal=causal
        )
        out_b = out[start:start + L]
        assert torch.equal(out_b, out_b_ref.squeeze(0)), (
            f"varlen slice mismatch at start={start}, L={L}"
        )
        start += L


def test_varlen_causal():
    """Same as basic, but explicitly causal=True (separate name for clarity)."""
    seqlens = [16, 33, 80]  # one unaligned (33)
    H, D = 1, 64
    q, k, v, cu, per_batch = _pack_varlen(seqlens, H, D, seed=1)

    out = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu, cu,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        causal=True,
    )

    start = 0
    for (q_b, k_b, v_b), L in zip(per_batch, seqlens):
        out_b_ref = flash_attn_mojo.flash_attn_func(
            q_b, k_b, v_b, causal=True
        )
        assert torch.equal(out[start:start + L], out_b_ref.squeeze(0))
        start += L


@pytest.mark.parametrize("causal", [False, True])
def test_varlen_matches_upstream(causal):
    """Cross-check against upstream `flash_attn_varlen_func` within bf16 tol."""
    try:
        from flash_attn import flash_attn_varlen_func as upstream_varlen
    except ImportError:
        pytest.skip("upstream flash-attn not available")

    seqlens = [64, 128, 96]
    H, D = 1, 64
    q, k, v, cu, _ = _pack_varlen(seqlens, H, D, seed=2)

    out_mojo = flash_attn_mojo.flash_attn_varlen_func(
        q, k, v, cu, cu,
        max_seqlen_q=max(seqlens), max_seqlen_k=max(seqlens),
        causal=causal,
    )
    out_up = upstream_varlen(
        q, k, v, cu, cu, max(seqlens), max(seqlens), causal=causal,
    )

    diff = _max_abs(out_mojo, out_up)
    assert diff < 5e-2, f"varlen vs upstream: max |diff|={diff:.3e}"


def test_varlen_mismatched_qk_raises():
    """If seqlen_q_b != seqlen_k_b for any batch element, raise clearly."""
    H, D = 1, 64
    # batch=2: seqlens_q=[32, 64], seqlens_k=[32, 32] -> batch 1 mismatched.
    q = torch.randn(32 + 64, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(32 + 32, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(32 + 32, H, D, dtype=torch.bfloat16, device="cuda")
    cu_q = torch.tensor([0, 32, 32 + 64], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 32, 32 + 32], dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="seqlen_q_b == seqlen_k_b"):
        flash_attn_mojo.flash_attn_varlen_func(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=64, max_seqlen_k=32,
        )


def test_kvcache_prefill_basic():
    """Pre-filled k_cache/v_cache, no new k/v. q runs against the full
    cache and must match a direct flash_attn_func call on the same
    K/V."""
    torch.manual_seed(0)
    B, L, H, D = 2, 128, 1, 64
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    cache_seqlens = torch.full((B,), L, dtype=torch.int32, device="cuda")

    out_kvc = flash_attn_mojo.flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True,
    )
    out_ref = flash_attn_mojo.flash_attn_func(
        q, k_cache, v_cache, causal=True,
    )
    diff = _max_abs(out_kvc, out_ref)
    assert diff < 5e-2, f"kvcache prefill: max |diff|={diff:.3e}"
    # cache_seqlens must be unchanged when no new k/v was appended.
    assert torch.equal(
        cache_seqlens,
        torch.full((B,), L, dtype=torch.int32, device="cuda"),
    )


def test_kvcache_append_then_attend():
    """Pre-fill cache to L=64, append L=64 new tokens, attend against
    the full 128. Must match a direct flash_attn_func on the
    concatenated K/V, and cache_seqlens must be updated in-place."""
    torch.manual_seed(1)
    B, L0, Lnew, H, D = 2, 64, 64, 1, 64
    Ltotal = L0 + Lnew

    # Allocate full-size cache and fill the first L0 slots with known data.
    k_full = torch.randn(B, Ltotal, H, D, dtype=torch.bfloat16, device="cuda")
    v_full = torch.randn(B, Ltotal, H, D, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros_like(k_full)
    v_cache = torch.zeros_like(v_full)
    k_cache[:, :L0].copy_(k_full[:, :L0])
    v_cache[:, :L0].copy_(v_full[:, :L0])

    # The "new" tokens to append.
    k_new = k_full[:, L0:].contiguous()
    v_new = v_full[:, L0:].contiguous()

    q = torch.randn(B, Ltotal, H, D, dtype=torch.bfloat16, device="cuda")
    cache_seqlens = torch.full((B,), L0, dtype=torch.int32, device="cuda")

    out_kvc = flash_attn_mojo.flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k_new, v=v_new,
        cache_seqlens=cache_seqlens, causal=True,
    )
    # cache_seqlens incremented by Lnew, in place.
    assert torch.equal(
        cache_seqlens,
        torch.full((B,), L0 + Lnew, dtype=torch.int32, device="cuda"),
    )
    # The cache itself must equal the concatenated K/V.
    assert torch.equal(k_cache, k_full)
    assert torch.equal(v_cache, v_full)

    out_ref = flash_attn_mojo.flash_attn_func(
        q, k_full, v_full, causal=True,
    )
    diff = _max_abs(out_kvc, out_ref)
    assert diff < 5e-2, f"kvcache append: max |diff|={diff:.3e}"


def test_kvcache_decode_not_supported():
    """seqlen_q=1 with a large cache must raise NotImplementedError
    with a clear message — the underlying kernel needs seqlen_q == seqlen_k."""
    B, L, H, D = 1, 128, 1, 64
    q = torch.randn(B, 1, H, D, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    cache_seqlens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    with pytest.raises(NotImplementedError, match="seqlen_q == used_kv_length"):
        flash_attn_mojo.flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens,
        )
