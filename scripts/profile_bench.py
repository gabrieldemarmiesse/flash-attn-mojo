"""One-shot bench harness for kernel profiling.

Runs fwd or bwd at a single (B, L, H, D) shape with a warmup phase
followed by a capture phase that is bracketed by
`cudaProfilerStart` / `cudaProfilerStop`. Combined with ncu's
`--profile-from-start no`, this lets the profiler skip every launch
that happens during JIT, allocator warmup, and torch's first-call
overhead — only the post-warmup `--iters` launches are recorded.

Usage (standalone — runs the bench, no profiling):
    uv run python scripts/profile_bench.py --kind bwd --shape 1,1024,8,64

Usage (under ncu):
    scripts/profile_kernel.sh --kind bwd --shape 1,1024,8,64
    # internally invokes:
    #   ncu --profile-from-start no --kernel-name regex:bwd \
    #     uv run python scripts/profile_bench.py --kind bwd \
    #       --shape 1,1024,8,64 --profile
"""

from __future__ import annotations

import argparse

import torch

import flash_attn_mojo


def _parse_shape(s: str) -> tuple[int, int, int, int]:
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--shape must be B,L,H,D")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["fwd", "bwd"], required=True)
    p.add_argument(
        "--shape",
        type=_parse_shape,
        default=(1, 1024, 8, 64),
        help="B,L,H,D (default: 1,1024,8,64)",
    )
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--iters",
        type=int,
        default=1,
        help="captured iterations (each runs fwd+bwd for --kind=bwd)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="bracket capture iters with cudaProfilerStart/Stop "
        "(needed when ncu is invoked with --profile-from-start no)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    B, L, H, D = args.shape
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)

    requires_grad = args.kind == "bwd"
    q = torch.randn(B, L, H, D, dtype=dtype, device="cuda", requires_grad=requires_grad)
    k = torch.randn(B, L, H, D, dtype=dtype, device="cuda", requires_grad=requires_grad)
    v = torch.randn(B, L, H, D, dtype=dtype, device="cuda", requires_grad=requires_grad)
    do = torch.randn_like(q) if args.kind == "bwd" else None

    def step() -> None:
        out = flash_attn_mojo.flash_attn_func(q, k, v, causal=args.causal)
        if args.kind == "bwd":
            out.backward(do)
            q.grad = None
            k.grad = None
            v.grad = None

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
