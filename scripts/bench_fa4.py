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


def _fa4_fwd_fn():
    from flash_attn.cute import flash_attn_func

    def run(q, k, v):
        return flash_attn_func(q, k, v, return_lse=True)  # (out, lse)

    return run


def _get_step(impl: str, kind: str, q, k, v, dout):
    """Returns (step_fn, outputs_fn). outputs_fn() -> tensors for the
    correctness check."""
    if kind == "fwd":
        if impl == "fa4":
            run = _fa4_fwd_fn()
        else:
            from flash_attn_mojo.fwd_fa4 import fa4_fwd as run

        out_holder = {}

        def step():
            out_holder["out"] = run(q, k, v)

        return step, lambda: out_holder["out"]  # (out, lse)

    # bwd: produce out/lse once with FA4's fwd (identical inputs for
    # both impls); time only the bwd path.
    from flash_attn.cute import flash_attn_func

    out, lse = flash_attn_func(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    grads = {}

    if impl == "fa4":
        from flash_attn.cute.interface import _flash_attn_bwd

        def step():
            dq, dk, dv = _flash_attn_bwd(q, k, v, out, dout, lse)
            grads["g"] = (dq, dk, dv)

    else:
        from flash_attn_mojo.bwd_fa4 import bwd_fa4

        def step():
            grads["g"] = bwd_fa4(q, k, v, out, dout, lse)

    return step, lambda: grads["g"]


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


def _check_small(impl: str, kind: str, D: int, seqlen: int = 512) -> None:
    torch.manual_seed(1)
    qs = torch.randn(2, seqlen, 4, D, dtype=torch.bfloat16, device="cuda")
    ks, vs = torch.randn_like(qs), torch.randn_like(qs)
    if kind == "fwd":
        step, outputs = _get_step(impl, "fwd", qs, ks, vs, None)
        step()
        out, lse = outputs()
        d = (out.float() - _sdpa_fp32(qs, ks, vs)).abs().max().item()
        scale = qs.shape[-1] ** -0.5
        ref_lse = torch.logsumexp(
            torch.einsum(
                "bshd,bthd->bhst", qs.float(), ks.float()
            ) * scale,
            dim=-1,
        )
        dl = (lse.float() - ref_lse).abs().max().item()
        print(
            f"CHECK impl={impl} kind=fwd S={seqlen} "
            f"small_vs_fp32_sdpa_maxdiff={d:.3e} lse_maxdiff={dl:.3e}"
        )
        return
    dos = torch.randn_like(qs)
    step, outputs = _get_step(impl, "bwd", qs, ks, vs, dos)
    step()
    qf = qs.detach().float().requires_grad_()
    kf = ks.detach().float().requires_grad_()
    vf = vs.detach().float().requires_grad_()
    import torch.nn.functional as F

    of = F.scaled_dot_product_attention(
        qf.transpose(1, 2), kf.transpose(1, 2), vf.transpose(1, 2)
    ).transpose(1, 2)
    of.backward(dos.float())
    names = ("dq", "dk", "dv")
    refs = (qf.grad, kf.grad, vf.grad)
    worst = 0.0
    for name, got, ref in zip(names, outputs(), refs):
        d = (got.float() - ref).abs().max().item()
        worst = max(worst, d)
        print(f"CHECK impl={impl} kind=bwd {name}_vs_fp32_maxdiff={d:.3e}")
    if worst > 5e-2:
        print("CHECK FAILED: bwd grads diverge from fp32 reference", file=sys.stderr)
        sys.exit(1)


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
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    B, S, H, D = args.shape
    torch.manual_seed(args.seed)

    if args.check_only:
        # 640 = 8 * 80 exercises the bwd tile_m=80 exact-fit path;
        # the others all leave a partial tail m-tile (S % 80 != 0).
        for s_len in (128, 256, 640, 1024):
            _check_small(args.impl, args.kind, D, seqlen=s_len)
        return

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dout = torch.randn_like(q) if args.kind == "bwd" else None

    step, outputs = _get_step(args.impl, args.kind, q, k, v, dout)

    # JIT compile + allocator warmup.
    step()
    torch.cuda.synchronize()

    if args.check:
        _check_small(args.impl, args.kind, D)
        if args.impl == "mojo":
            ref_step, ref_outputs = _get_step("fa4", args.kind, q, k, v, dout)
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

    fwd_flops = 4 * B * H * S * S * D
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
    print(
        f"RESULT impl={args.impl} kind={args.kind} shape={B},{S},{H},{D} "
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
