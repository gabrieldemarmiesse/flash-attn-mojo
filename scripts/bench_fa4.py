"""Bench harness for the FA4-vs-Mojo forward race.

Measures kernel-only GPU time (torch.profiler / CUPTI) of one
implementation at one shape, prints a machine-parseable RESULT line.

Implementations:
    fa4   Tri Dao FlashAttention-4 (flash_attn.cute) — the target.
    mojo  flash_attn_mojo.fwd_fa4 — the kernel being developed.

Usage:
    uv run python scripts/bench_fa4.py --impl mojo --shape 2,8192,16,128
    uv run python scripts/bench_fa4.py --impl fa4 --check

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


def _get_impl(name: str):
    if name == "fa4":
        from flash_attn.cute import flash_attn_func

        def run(q, k, v):
            out, _lse = flash_attn_func(q, k, v)
            return out

        return run
    if name == "mojo":
        from flash_attn_mojo.fwd_fa4 import fa4_fwd

        return fa4_fwd
    raise ValueError(name)


def _sdpa_ref_fp32(q, k, v):
    import torch.nn.functional as F

    return (
        F.scaled_dot_product_attention(
            q.transpose(1, 2).float(),
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
        )
        .transpose(1, 2)
        .to(q.dtype)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--impl", choices=["fa4", "mojo"], required=True)
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
        help="correctness: small-shape vs fp32 SDPA + bench-shape vs fa4",
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
    run = _get_impl(args.impl)

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # JIT compile + allocator warmup.
    out = run(q, k, v)
    torch.cuda.synchronize()

    if args.check:
        qs = torch.randn(2, 512, 4, D, dtype=torch.bfloat16, device="cuda")
        ks, vs = torch.randn_like(qs), torch.randn_like(qs)
        d_small = (run(qs, ks, vs).float() - _sdpa_ref_fp32(qs, ks, vs).float()).abs().max().item()
        print(f"CHECK impl={args.impl} small_vs_fp32_sdpa_maxdiff={d_small:.3e}")
        if args.impl == "mojo":
            ref = _get_impl("fa4")(q, k, v)
            d_big = (out - ref).abs().max().item()
            print(f"CHECK impl=mojo bench_shape_vs_fa4_maxdiff={d_big:.3e}")
            if d_big > 5e-2:
                print("CHECK FAILED: mojo output diverges from fa4", file=sys.stderr)
                sys.exit(1)

    for _ in range(args.warmup):
        run(q, k, v)
    torch.cuda.synchronize()

    flops = 4 * B * H * S * S * D

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.iters):
            run(q, k, v)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            run(q, k, v)
        torch.cuda.synchronize()

    kernel_us = 0.0
    for e in prof.key_averages():
        name = e.key.lower()
        if e.device_time_total > 0 and "memcpy" not in name and "memset" not in name:
            kernel_us += e.device_time_total
    us = kernel_us / args.iters
    tflops = flops / (us * 1e-6) / 1e12
    print(
        f"RESULT impl={args.impl} shape={B},{S},{H},{D} "
        f"us={us:.1f} tflops={tflops:.1f}"
    )


if __name__ == "__main__":
    main()
