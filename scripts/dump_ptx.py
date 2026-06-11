"""Dump exactly one kernel variant's PTX (no pytest delegation).

`bench_fa4.py --check-only` delegates to pytest, which may compile
several variants in one process — with MOJO_DUMP_PTX set, the LAST
compile wins the dump file. This script JITs precisely the requested
variant via the wrapper functions on tiny tensors.

Usage:
    MOJO_DUMP_PTX=ptx/x.ptx uv run python scripts/dump_ptx.py \
        --kind fwd [--causal] [--hkv N] [--varlen] [--dtype bf16] \
        [--hdim 128]
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["fwd", "bwd"], default="fwd")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--varlen", action="store_true")
    p.add_argument("--window", type=int, default=0,
                   help="window_left (fwd, implies --causal semantics)")
    p.add_argument("--softcap", type=float, default=0.0)
    p.add_argument("--hkv", type=int, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--hdim", type=int, default=128)
    args = p.parse_args()

    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    D = args.hdim
    H = 4
    h_kv = args.hkv if args.hkv else H
    S = 384 if D == 64 else 256

    if args.varlen:
        cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
        q = torch.randn(S, H, D, dtype=dt, device="cuda")
        k = torch.randn(S, h_kv, D, dtype=dt, device="cuda")
        v = torch.randn_like(k)
        from flash_attn_mojo.fwd_fa4 import fa4_varlen_fwd

        out, lse = fa4_varlen_fwd(q, k, v, cu, cu, causal=args.causal)
        if args.kind == "bwd":
            from flash_attn_mojo.bwd_fa4 import bwd_fa4_varlen

            bwd_fa4_varlen(
                q, k, v, out, torch.randn_like(q), lse, cu, cu,
                causal=args.causal,
            )
    else:
        q = torch.randn(2, S, H, D, dtype=dt, device="cuda")
        k = torch.randn(2, S, h_kv, D, dtype=dt, device="cuda")
        v = torch.randn_like(k)
        from flash_attn_mojo.fwd_fa4 import fa4_fwd

        out, lse = fa4_fwd(
            q, k, v, causal=args.causal, window_left=args.window,
            softcap=args.softcap,
        )
        if args.kind == "bwd":
            from flash_attn_mojo.bwd_fa4 import bwd_fa4

            bwd_fa4(
                q, k, v, out, torch.randn_like(q), lse,
                causal=args.causal, window_left=args.window,
                softcap=args.softcap,
            )
    torch.cuda.synchronize()
    print("dumped", args)


if __name__ == "__main__":
    main()
