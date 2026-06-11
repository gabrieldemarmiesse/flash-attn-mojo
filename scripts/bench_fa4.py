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


def _fa4_fwd_fn(
    causal: bool, window_left: int = 0, softcap: float = 0.0
):
    from flash_attn.cute import flash_attn_func

    win = {"window_size": (window_left, 0)} if window_left else {}
    if softcap:
        win["softcap"] = softcap

    def run(q, k, v):
        return flash_attn_func(
            q, k, v, causal=causal, return_lse=True, **win
        )  # (out, lse)

    return run


def _get_step(
    impl: str, kind: str, q, k, v, dout, causal: bool = False,
    window_left: int = 0, softcap: float = 0.0,
):
    """Returns (step_fn, outputs_fn). outputs_fn() -> tensors for the
    correctness check."""
    if kind == "fwd":
        if impl == "fa4":
            run = _fa4_fwd_fn(causal, window_left, softcap)
        else:
            from flash_attn_mojo.fwd_fa4 import fa4_fwd as _mojo_fwd

            scap = {"softcap": softcap} if softcap else {}

            def run(q, k, v):
                return _mojo_fwd(
                    q, k, v, causal=causal, window_left=window_left,
                    **scap,
                )

        out_holder = {}

        def step():
            out_holder["out"] = run(q, k, v)

        return step, lambda: out_holder["out"]  # (out, lse)

    # bwd: produce out/lse once with FA4's fwd (identical inputs for
    # both impls); time only the bwd path.
    from flash_attn.cute import flash_attn_func

    win = {"window_size": (window_left, 0)} if window_left else {}
    bwin = {"window_size_left": window_left} if window_left else {}
    if softcap:
        win["softcap"] = softcap
        bwin["softcap"] = softcap
    out, lse = flash_attn_func(
        q, k, v, causal=causal, return_lse=True, **win
    )
    torch.cuda.synchronize()
    grads = {}

    if impl == "fa4":
        from flash_attn.cute.interface import _flash_attn_bwd

        def step():
            dq, dk, dv = _flash_attn_bwd(
                q, k, v, out, dout, lse, causal=causal, **bwin
            )
            grads["g"] = (dq, dk, dv)

    else:
        from flash_attn_mojo.bwd_fa4 import bwd_fa4

        mwin = {"window_left": window_left} if window_left else {}
        if softcap:
            mwin["softcap"] = softcap

        def step():
            grads["g"] = bwd_fa4(
                q, k, v, out, dout, lse, causal=causal, **mwin
            )

    return step, lambda: grads["g"]


def _run_check_suite(args) -> None:
    """Correctness checks live in tests/test_kernels.py; delegate to
    pytest with a -k expression selecting this variant (the test and
    parameter names encode the kind/dense-varlen/mask/heads axes).
    FLASH_ATTN_MOJO_TEST_IMPL carries the impl under test."""
    import os
    import subprocess
    from pathlib import Path

    if args.softcap:
        # softcap tests are their own axis (plain/causal/swa x
        # mha/gqa at hd128).
        parts = [args.kind, "softcap"]
    elif args.window:
        # window tests are their own axis (always causal+mha+hd128).
        parts = [args.kind, "window"]
    else:
        parts = [
            args.kind,
            "varlen" if args.varlen else "dense",
            "causal" if args.causal else "plain",
        ]
        if not args.varlen:
            parts.append("gqa" if args.hkv else "mha")
        # dtype axis: bf16 selects bf16-id'd + un-id'd (canonical)
        # tests; fp16 selects only the fp16-id'd ones. Same for hdim.
        parts.append("not fp16" if args.dtype == "bf16" else "fp16")
        parts.append("hd64" if args.shape[3] == 64 else "not hd64")
    kexpr = " and ".join(parts)
    tests = Path(__file__).resolve().parent.parent / "tests" / "test_kernels.py"
    print(f"CHECK pytest -k '{kexpr}' (impl={args.impl})")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests), "-q", "-k", kexpr],
        env={**os.environ, "FLASH_ATTN_MOJO_TEST_IMPL": args.impl},
    )
    if r.returncode != 0:
        print("CHECK FAILED (see pytest output above)", file=sys.stderr)
        sys.exit(r.returncode)



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
        help="run this variant's pytest checks (tests/test_kernels.py)"
        " before benching",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="run this variant's pytest checks and exit (fast "
        "edit-compile-check loop; no bench)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="no timing; bracket capture iters with cudaProfilerStart/Stop",
    )
    p.add_argument("--causal", action="store_true")
    p.add_argument(
        "--window", type=int, default=0,
        help="sliding-window left width (causal local attention); "
        "0 = no window",
    )
    p.add_argument(
        "--softcap", type=float, default=0.0,
        help="attention logit softcap (Gemma-2 style); 0 = off",
    )
    p.add_argument(
        "--dtype", choices=["bf16", "fp16"], default="bf16",
        help="tensor dtype for the bench shapes",
    )
    p.add_argument(
        "--varlen",
        action="store_true",
        help="packed varlen (cu_seqlens) mode",
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
    DT = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    if args.check_only:
        _run_check_suite(args)
        return
    if args.check:
        _run_check_suite(args)

    if args.varlen:
        # Packed varlen bench: one (total, H, D) packed batch.
        lens = [int(x) for x in args.varlen_lens.split(",")]
        cu_list = [0]
        for L in lens:
            cu_list.append(cu_list[-1] + L)
        total = cu_list[-1]
        cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
        q = torch.randn(total, H, D, dtype=DT, device="cuda")
        h_kv = args.hkv if args.hkv else H
        k = torch.randn(total, h_kv, D, dtype=DT, device="cuda")
        v = torch.randn_like(k)
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
        q = torch.randn(B, S, H, D, dtype=DT, device="cuda")
        h_kv = args.hkv if args.hkv else H
        k = torch.randn(B, S, h_kv, D, dtype=DT, device="cuda")
        v = torch.randn_like(k)
        dout = torch.randn_like(q) if args.kind == "bwd" else None

        step, outputs = _get_step(
            args.impl, args.kind, q, k, v, dout, args.causal,
            args.window, args.softcap,
        )

    # JIT compile + allocator warmup.
    step()
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    if args.varlen:
        fwd_flops = 4 * H * D * sum(L * L for L in lens)
        if args.causal:
            fwd_flops //= 2
    elif args.window:
        # causal local: row i attends min(i+1, window+1) keys.
        W = args.window
        attended = S * (W + 1) - W * (W + 1) // 2 if S > W else S * (S + 1) // 2
        fwd_flops = 4 * B * H * D * attended
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
        hkv_str = str(h_kv)
    else:
        shape_str = f"{B},{S},{H},{D}"
        hkv_str = str(h_kv)
    print(
        f"RESULT impl={args.impl} kind={args.kind} "
        f"shape={shape_str} causal={int(args.causal)} "
        f"hkv={hkv_str} dtype={args.dtype} window={args.window} "
        f"softcap={args.softcap:g} "
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
