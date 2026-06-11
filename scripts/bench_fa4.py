"""Bench harness for the FA4-vs-Mojo race (fwd and bwd).

Measures kernel-only GPU time (torch.profiler / CUPTI) of one
implementation at one shape, prints a machine-parseable RESULT line.

Implementations:
    fa4   Tri Dao FlashAttention-4 (flash_attn.cute) — the target.
    mojo  flash_attn_mojo.{fwd_fa4,bwd_fa4} — the kernels being
          developed.

Kinds:
    fwd   out = attn(q, k, v)
    bwd   dq, dk, dv = attn_bwd(q, k, v, out, dout, lse). The fwd
          (always FA4's, for identical out/lse inputs) runs outside
          the timed region; only the bwd kernels are measured.

Usage:
    uv run python scripts/bench_fa4.py --impl mojo --kind bwd
    uv run python scripts/bench_fa4.py --impl fa4 --kind fwd --check

Under ncu (bracketed capture, see master_bench.sh):
    ncu --profile-from-start no ... \
        uv run python scripts/bench_fa4.py --impl mojo --profile --iters 1
"""

from __future__ import annotations

import argparse
import sys

import torch


def _parse_shape(s: str) -> tuple[int, int, int, int]:
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--shape must be B,S,H,D")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def _fa4_fwd_fn(causal: bool):
    from flash_attn.cute import flash_attn_func

    def run(q, k, v):
        return flash_attn_func(
            q, k, v, causal=causal, return_lse=True
        )  # (out, lse)

    return run


def _get_step(impl: str, kind: str, q, k, v, dout, causal: bool = False):
    """Returns (step_fn, outputs_fn). outputs_fn() -> tensors for the
    correctness check."""
    if kind == "fwd":
        if impl == "fa4":
            run = _fa4_fwd_fn(causal)
        else:
            from flash_attn_mojo.fwd_fa4 import fa4_fwd as _mojo_fwd

            def run(q, k, v):
                return _mojo_fwd(q, k, v, causal=causal)

        out_holder = {}

        def step():
            out_holder["out"] = run(q, k, v)

        return step, lambda: out_holder["out"]  # (out, lse)

    # bwd: produce out/lse once with FA4's fwd (identical inputs for
    # both impls); time only the bwd path.
    from flash_attn.cute import flash_attn_func

    out, lse = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    torch.cuda.synchronize()
    grads = {}

    if impl == "fa4":
        from flash_attn.cute.interface import _flash_attn_bwd

        def step():
            dq, dk, dv = _flash_attn_bwd(
                q, k, v, out, dout, lse, causal=causal
            )
            grads["g"] = (dq, dk, dv)

    else:
        from flash_attn_mojo.bwd_fa4 import bwd_fa4

        def step():
            grads["g"] = bwd_fa4(q, k, v, out, dout, lse, causal=causal)

    return step, lambda: grads["g"]


def _rep_kv(t, nheads_q):
    """repeat_interleave KV heads up to nheads_q (GQA reference)."""
    g = nheads_q // t.shape[2]
    return t.repeat_interleave(g, dim=2) if g > 1 else t


def _sdpa_fp32(q, k, v, causal: bool = False):
    import torch.nn.functional as F

    return (
        F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            _rep_kv(k, q.shape[2]).transpose(1, 2).float(),
            _rep_kv(v, q.shape[2]).transpose(1, 2).float(),
            is_causal=causal,
        )
        .transpose(1, 2)
    )


def _check_small(
    impl: str,
    kind: str,
    D: int,
    seqlen: int = 512,
    causal: bool = False,
    hkv: int = 0,
) -> None:
    torch.manual_seed(1)
    hq = 4
    qs = torch.randn(2, seqlen, hq, D, dtype=torch.bfloat16, device="cuda")
    h_kv = hkv if hkv else hq
    # small-shape GQA uses ratio Hq/Hkv = 4/2 when --hkv is set
    if hkv:
        h_kv = 2
    ks = torch.randn(2, seqlen, h_kv, D, dtype=torch.bfloat16, device="cuda")
    vs = torch.randn_like(ks)
    if kind == "fwd":
        step, outputs = _get_step(impl, "fwd", qs, ks, vs, None, causal)
        step()
        out, lse = outputs()
        d = (out.float() - _sdpa_fp32(qs, ks, vs, causal)).abs().max().item()
        scale = qs.shape[-1] ** -0.5
        scores = (
            torch.einsum(
                "bshd,bthd->bhst",
                qs.float(),
                _rep_kv(ks, qs.shape[2]).float(),
            )
            * scale
        )
        if causal:
            s_q = scores.shape[-2]
            mask = torch.ones(
                s_q, s_q, dtype=torch.bool, device=scores.device
            ).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        ref_lse = torch.logsumexp(
            scores,
            dim=-1,
        )
        dl = (lse.float() - ref_lse).abs().max().item()
        print(
            f"CHECK impl={impl} kind=fwd S={seqlen} causal={int(causal)} "
            f"small_vs_fp32_sdpa_maxdiff={d:.3e} lse_maxdiff={dl:.3e}"
        )
        return
    dos = torch.randn_like(qs)
    step, outputs = _get_step(impl, "bwd", qs, ks, vs, dos, causal)
    step()
    qf = qs.detach().float().requires_grad_()
    kf = ks.detach().float().requires_grad_()
    vf = vs.detach().float().requires_grad_()
    import torch.nn.functional as F

    of = F.scaled_dot_product_attention(
        qf.transpose(1, 2),
        _rep_kv(kf, qf.shape[2]).transpose(1, 2),
        _rep_kv(vf, qf.shape[2]).transpose(1, 2),
        is_causal=causal,
    ).transpose(1, 2)
    of.backward(dos.float())
    names = ("dq", "dk", "dv")
    refs = (qf.grad, kf.grad, vf.grad)
    worst = 0.0
    for name, got, ref in zip(names, outputs(), refs):
        d = (got.float() - ref).abs().max().item()
        worst = max(worst, d)
        print(
            f"CHECK impl={impl} kind=bwd causal={int(causal)} "
            f"{name}_vs_fp32_maxdiff={d:.3e}"
        )
    if worst > 5e-2:
        print("CHECK FAILED: bwd grads diverge from fp32 reference", file=sys.stderr)
        sys.exit(1)


def _check_varlen(
    impl: str,
    kind: str,
    D: int,
    lens: list[int],
    causal: bool = False,
) -> None:
    """Packed-varlen correctness vs per-sequence fp32 references:
    fwd checks out + LSE (FA4's packed (H, total_q) layout); bwd
    checks dq/dk/dv vs fp32 SDPA autograd."""
    torch.manual_seed(3)
    H = 4
    cu_list = [0]
    for L in lens:
        cu_list.append(cu_list[-1] + L)
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    total = cu_list[-1]
    q = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if kind == "bwd":
        # out/lse always from FA4's varlen fwd (identical inputs for
        # both impls); only the bwd path differs.
        from flash_attn.cute import flash_attn_varlen_func

        out, lse = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max(lens), max_seqlen_k=max(lens),
            causal=causal, return_lse=True,
        )
        dout = torch.randn_like(q)
        if impl == "mojo":
            from flash_attn_mojo.bwd_fa4 import bwd_fa4_varlen

            dq, dk, dv = bwd_fa4_varlen(
                q, k, v, out, dout, lse, cu, cu, causal=causal
            )
        else:
            from flash_attn.cute.interface import _flash_attn_bwd

            dq, dk, dv = _flash_attn_bwd(
                q, k, v, out, dout, lse,
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                causal=causal,
            )
        worst = {"dq": 0.0, "dk": 0.0, "dv": 0.0}
        for i, L in enumerate(lens):
            s, e = cu_list[i], cu_list[i + 1]
            qf = q[s:e].detach().float().requires_grad_()
            kf = k[s:e].detach().float().requires_grad_()
            vf = v[s:e].detach().float().requires_grad_()
            import torch.nn.functional as F

            of = F.scaled_dot_product_attention(
                qf.transpose(0, 1),
                kf.transpose(0, 1),
                vf.transpose(0, 1),
                is_causal=causal,
            ).transpose(0, 1)
            of.backward(dout[s:e].float())
            for name, got, ref in (
                ("dq", dq[s:e], qf.grad),
                ("dk", dk[s:e], kf.grad),
                ("dv", dv[s:e], vf.grad),
            ):
                worst[name] = max(
                    worst[name],
                    (got.float() - ref).abs().max().item(),
                )
        print(
            f"CHECK impl={impl} kind=bwd "
            f"varlen={','.join(map(str, lens))} causal={int(causal)} "
            f"dq_maxdiff={worst['dq']:.3e} dk_maxdiff={worst['dk']:.3e} "
            f"dv_maxdiff={worst['dv']:.3e}"
        )
        if max(worst.values()) > 5e-2:
            print(
                "CHECK FAILED: varlen bwd grads diverge from fp32 "
                "reference",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    if impl == "mojo":
        from flash_attn_mojo.fwd_fa4 import fa4_varlen_fwd

        out, lse = fa4_varlen_fwd(q, k, v, cu, cu, causal=causal)
    else:
        from flash_attn.cute import flash_attn_varlen_func

        out, lse = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max(lens), max_seqlen_k=max(lens),
            causal=causal, return_lse=True,
        )

    scale = D**-0.5
    worst = worst_lse = 0.0
    for i, L in enumerate(lens):
        s, e = cu_list[i], cu_list[i + 1]
        ref = _sdpa_fp32(
            q[s:e].unsqueeze(0), k[s:e].unsqueeze(0), v[s:e].unsqueeze(0),
            causal,
        )[0]
        d = (out[s:e].float() - ref).abs().max().item()
        worst = max(worst, d)
        scores = (
            torch.einsum("shd,thd->hst", q[s:e].float(), k[s:e].float())
            * scale
        )
        if causal:
            mask = torch.ones(
                L, L, dtype=torch.bool, device=scores.device
            ).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        ref_lse = torch.logsumexp(scores, dim=-1)  # (H, L)
        dl = (lse[:, s:e].float() - ref_lse).abs().max().item()
        worst_lse = max(worst_lse, dl)
    print(
        f"CHECK impl={impl} kind=fwd varlen={','.join(map(str, lens))} "
        f"causal={int(causal)} vs_fp32_sdpa_maxdiff={worst:.3e} "
        f"lse_maxdiff={worst_lse:.3e}"
    )
    if worst > 5e-2 or worst_lse > 1e-3:
        print(
            "CHECK FAILED: varlen fwd diverges from fp32 reference",
            file=sys.stderr,
        )
        sys.exit(1)


# Tile-aligned varlen check sets (every length % 128 == 0); ragged
# lengths join when the seqlen tail masking lands (port step 2).
_VARLEN_CHECK_SETS = (
    [128],
    [128, 256, 640],
    [1024, 128],
    [256] * 16,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--impl", choices=["fa4", "mojo"], required=True)
    p.add_argument("--kind", choices=["fwd", "bwd"], default="fwd")
    p.add_argument(
        "--shape",
        type=_parse_shape,
        default=(2, 8192, 16, 128),
        help="B,S,H,D (default: canonical 2,8192,16,128)",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--check",
        action="store_true",
        help="correctness: small-shape vs fp32 + bench-shape vs fa4",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="run the small-shape fp32 checks at a few seqlens and "
        "exit (fast edit-compile-check loop; no bench)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="no timing; bracket capture iters with cudaProfilerStart/Stop",
    )
    p.add_argument("--causal", action="store_true")
    p.add_argument(
        "--varlen",
        action="store_true",
        help="packed varlen (cu_seqlens) mode; with --check-only runs "
        "the varlen check sets",
    )
    p.add_argument(
        "--varlen-lens",
        default="3072,2816,2560,2048,1792,1536,1280,1280",
        help="comma-separated per-sequence lengths for the varlen "
        "bench (default: canonical 8-seq mixed config, 16384 tokens)",
    )
    p.add_argument(
        "--hkv",
        type=int,
        default=0,
        help="KV heads (GQA); 0 = same as Hq (MHA)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    B, S, H, D = args.shape
    torch.manual_seed(args.seed)

    if args.check_only and args.varlen:
        for lens in _VARLEN_CHECK_SETS:
            _check_varlen(
                args.impl, args.kind, D, lens, causal=args.causal
            )
        return

    if args.check_only:
        # 640 = 8 * 80 exercises the bwd tile_m=80 exact-fit path;
        # the others all leave a partial tail m-tile (S % 80 != 0).
        for s_len in (128, 256, 640, 1024):
            _check_small(
                args.impl, args.kind, D, seqlen=s_len,
                causal=args.causal, hkv=args.hkv,
            )
        return

    if args.varlen:
        # Packed varlen bench: one (total, H, D) packed batch.
        lens = [int(x) for x in args.varlen_lens.split(",")]
        cu_list = [0]
        for L in lens:
            cu_list.append(cu_list[-1] + L)
        total = cu_list[-1]
        cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
        q = torch.randn(total, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        holder = {}
        if args.kind == "bwd":
            # out/lse from FA4's varlen fwd, outside the timed region.
            from flash_attn.cute import flash_attn_varlen_func

            out, lse = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                causal=args.causal, return_lse=True,
            )
            torch.cuda.synchronize()
            dout = torch.randn_like(q)
            if args.impl == "mojo":
                from flash_attn_mojo.bwd_fa4 import bwd_fa4_varlen

                def step():
                    holder["o"] = bwd_fa4_varlen(
                        q, k, v, out, dout, lse, cu, cu,
                        causal=args.causal,
                    )

            else:
                from flash_attn.cute.interface import _flash_attn_bwd

                def step():
                    holder["o"] = _flash_attn_bwd(
                        q, k, v, out, dout, lse,
                        cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                        causal=args.causal,
                    )

        elif args.impl == "mojo":
            from flash_attn_mojo.fwd_fa4 import fa4_varlen_fwd

            def step():
                holder["o"] = fa4_varlen_fwd(
                    q, k, v, cu, cu, causal=args.causal
                )

        else:
            from flash_attn.cute import flash_attn_varlen_func

            def step():
                holder["o"] = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                    causal=args.causal, return_lse=True,
                )

        outputs = lambda: holder["o"]  # noqa: E731
    else:
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        h_kv = args.hkv if args.hkv else H
        k = torch.randn(B, S, h_kv, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        dout = torch.randn_like(q) if args.kind == "bwd" else None

        step, outputs = _get_step(
            args.impl, args.kind, q, k, v, dout, args.causal
        )

    # JIT compile + allocator warmup.
    step()
    torch.cuda.synchronize()

    if args.check and args.varlen:
        for check_lens in _VARLEN_CHECK_SETS:
            _check_varlen(
                args.impl, args.kind, D, check_lens, causal=args.causal
            )
        if args.impl == "mojo":
            if args.kind == "bwd":
                from flash_attn.cute.interface import _flash_attn_bwd

                refs = _flash_attn_bwd(
                    q, k, v, out, dout, lse,
                    cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                    causal=args.causal,
                )
                torch.cuda.synchronize()
                worst = max(
                    (got - ref).abs().max().item()
                    for got, ref in zip(outputs(), refs)
                )
            else:
                from flash_attn.cute import flash_attn_varlen_func

                ref_out, ref_lse = flash_attn_varlen_func(
                    q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=max(lens), max_seqlen_k=max(lens),
                    causal=args.causal, return_lse=True,
                )
                torch.cuda.synchronize()
                o_got, lse_got = outputs()
                worst = max(
                    (o_got - ref_out).abs().max().item(),
                    (lse_got - ref_lse).abs().max().item(),
                )
            print(
                f"CHECK impl=mojo kind={args.kind} varlen=1 "
                f"bench_shape_vs_fa4_maxdiff={worst:.3e}"
            )
            if worst > 5e-2:
                print(
                    "CHECK FAILED: mojo varlen output diverges from fa4",
                    file=sys.stderr,
                )
                sys.exit(1)
    elif args.check:
        _check_small(
            args.impl, args.kind, D, causal=args.causal, hkv=args.hkv
        )
        if args.impl == "mojo":
            ref_step, ref_outputs = _get_step(
                "fa4", args.kind, q, k, v, dout, args.causal
            )
            ref_step()
            torch.cuda.synchronize()
            worst = 0.0
            for got, ref in zip(outputs(), ref_outputs()):
                worst = max(worst, (got - ref).abs().max().item())
            print(
                f"CHECK impl=mojo kind={args.kind} "
                f"bench_shape_vs_fa4_maxdiff={worst:.3e}"
            )
            if worst > 5e-2:
                print(
                    "CHECK FAILED: mojo output diverges from fa4",
                    file=sys.stderr,
                )
                sys.exit(1)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    if args.varlen:
        fwd_flops = 4 * H * D * sum(L * L for L in lens)
    else:
        fwd_flops = 4 * B * H * S * S * D
    if args.causal:
        fwd_flops //= 2
    flops = fwd_flops if args.kind == "fwd" else fwd_flops * 5 // 2

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()

    kernel_us = 0.0
    for e in prof.key_averages():
        name = e.key.lower()
        if e.device_time_total > 0 and "memcpy" not in name and "memset" not in name:
            kernel_us += e.device_time_total
    us = kernel_us / args.iters
    tflops = flops / (us * 1e-6) / 1e12
    if args.varlen:
        shape_str = f"varlen:{total}tok,{H},{D}"
        hkv_str = str(H)
    else:
        shape_str = f"{B},{S},{H},{D}"
        hkv_str = str(h_kv)
    print(
        f"RESULT impl={args.impl} kind={args.kind} "
        f"shape={shape_str} causal={int(args.causal)} "
        f"hkv={hkv_str} "
        f"us={us:.1f} tflops={tflops:.1f}"
    )
    for e in sorted(
        prof.key_averages(), key=lambda e: -e.device_time_total
    ):
        name = e.key.lower()
        if e.device_time_total > 0 and "memcpy" not in name and "memset" not in name:
            short = e.key.split("(")[0][:60]
            print(
                f"KERNEL impl={args.impl} name={short} "
                f"us={e.device_time_total / args.iters:.1f}"
            )


if __name__ == "__main__":
    main()
