"""Reference bench for the AMD (ROCm / MI300X) flash-attention race.

The AMD analog of bench_fa4.py's reference lane. There is no CuTe /
FlashAttention-4 on AMD, so the reference baseline is Tri Dao's
`flash_attn` built with the **Composable Kernel (CK)** backend — the
fastest attention kernel on this MI300X (measured ranking CK > Triton >
SDPA). It is the ONLY reference here; CK must be installed (build it per
scripts/README.md) or this script exits with an error.

The Mojo lane is the standalone bench/bench_mojo_rocm.mojo binary
(wall-clock timed, its own JSON) — this script measures ONLY the CK
reference, kernel-only via torch.profiler (roctracer on ROCm, the CUPTI
analog), and prints a machine-parseable RESULT line matching
bench_fa4.py's format so master_bench.py can diff the two.

Usage:
    uv run python scripts/bench_rocm.py --seq 4096 --heads 16 \
        --head-dim 128 [--causal] [--check]
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def _get_step(q, k, v, causal: bool):
    """Returns (step_fn, out_fn) for the CK-flash reference. q/k/v are
    [B, H, S, D]. Exits if the CK backend is not installed."""
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        sys.exit(
            "flash_attn (Composable Kernel backend) is not installed — it is "
            "the required reference baseline. Build it per scripts/README.md."
        )

    # flash_attn wants [B, S, H, D].
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    holder = {}

    def step():
        holder["o"] = flash_attn_func(qt, kt, vt, causal=causal)

    return step, lambda: holder["o"].transpose(1, 2)


def _check(q, k, v, out, causal: bool) -> float:
    """Max abs error of `out` vs an fp32 SDPA reference (math backend)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=causal
        )
    return (out.float() - ref).abs().max().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument(
        "--dtype", choices=["fp16", "bf16"], default="fp16"
    )
    p.add_argument(
        "--walltime", action="store_true",
        help="CUDA-event end-to-end time instead of kernel-only",
    )
    p.add_argument("--check", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no ROCm/CUDA device visible to torch")

    torch.manual_seed(args.seed)
    DT = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    B, S, H, D = args.batch, args.seq, args.heads, args.head_dim
    q = torch.randn(B, H, S, D, dtype=DT, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    step, out_fn = _get_step(q, k, v, args.causal)

    # JIT / autotune warmup.
    step()
    torch.cuda.synchronize()

    if args.check:
        err = _check(q, k, v, out_fn(), args.causal)
        print(f"CHECK impl=ck max_abs_err={err:.3e}")
        if err > 5e-3:
            print("CHECK FAILED", file=sys.stderr)
            sys.exit(1)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    fwd_flops = 4 * B * H * S * S * D
    if args.causal:
        fwd_flops //= 2

    def emit(us: float, measure: str) -> None:
        tflops = fwd_flops / (us * 1e-6) / 1e12
        print(
            f"RESULT impl=ck kind=fwd "
            f"shape={B},{S},{H},{D} causal={int(args.causal)} "
            f"dtype={args.dtype} measure={measure} "
            f"us={us:.1f} tflops={tflops:.1f}"
        )

    if args.walltime:
        for _ in range(args.runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                step()
            end.record()
            torch.cuda.synchronize()
            emit(start.elapsed_time(end) * 1e3 / args.iters, "walltime")
        return

    from torch.profiler import ProfilerActivity, profile

    last = None
    for _ in range(args.runs):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                step()
            torch.cuda.synchronize()
        kernel_us = 0.0
        for e in prof.key_averages():
            name = e.key.lower()
            if (
                e.device_time_total > 0
                and "memcpy" not in name
                and "memset" not in name
            ):
                kernel_us += e.device_time_total
        emit(kernel_us / args.iters, "kernel")
        last = prof

    for e in sorted(last.key_averages(), key=lambda e: -e.device_time_total):
        name = e.key.lower()
        if e.device_time_total > 0 and "memcpy" not in name and "memset" not in name:
            short = e.key.split("(")[0][:60]
            print(
                f"KERNEL impl=ck name={short} "
                f"us={e.device_time_total / args.iters:.1f}"
            )


if __name__ == "__main__":
    main()
