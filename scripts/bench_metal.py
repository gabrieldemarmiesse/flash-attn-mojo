#!/usr/bin/env python3
"""Process-interleaved 3-way bench: mojo vs MFA vs ccv on the Apple GPU.

Each implementation is a small CLI with the same contract: it
generates its own seeded inputs, runs `--warmup` + `--iters` timed
forward-attention executions (each = `--dispatches` back-to-back
dispatches in one command buffer, kernel-only GPU time divided out),
and prints one JSON object on the last stdout line:

    {"gpu_time_us": [...], "check_max_error": <optional>, ...}

This orchestrator interleaves the CLIs round-robin (the unlocked-GPU
analog of the NVIDIA interleaved bench: drift hits everyone equally),
pools trials, and reports min/median, per-round-median spread, GFLOPS
(4*S*S*D), and mojo-vs-reference ratios (<1 means mojo is faster,
same convention as the FA4 race).

Single dispatch per command buffer flaps 5-50x on M4 (GPU power
states) — keep --dispatches >= 5.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Per-dtype max|err| gates vs each CLI's built-in fp32 reference check.
# (MFA measures ~1e-5 at fp16 inputs/intermediates; gates are loose
# enough for run-to-run input variation, tight enough to catch a
# broken mask/softmax, which shows up at >=1e-2.)
CHECK_TOL = {"fp16": 1e-3, "fp32": 5e-5, "bf16": 1e-2}


def mfa_cmd(a: argparse.Namespace) -> tuple[list[str], Path]:
    # MFA is single-head; the wrapper emulates MHA with one dispatch per
    # head per pass (per-head buffers -> heads run concurrently, repeats
    # serialize on each head's O/L hazard).
    bin_ = REPO / "reference_air/mfa/bench_mfa/.build/release/bench_mfa"
    cmd = [
        str(bin_),
        "--seq", str(a.seq),
        "--head-dim", str(a.head_dim),
        "--heads", str(a.heads),
        "--dtype", a.dtype,
        "--iters", str(a.iters),
        "--warmup", str(a.warmup),
        "--dispatches", str(a.dispatches),
    ]
    if a.check:
        cmd.append("--check")
    return cmd, bin_


def mfa_build() -> tuple[list[str], Path]:
    return ["swift", "build", "-c", "release"], REPO / "reference_air/mfa/bench_mfa"


def ccv_cmd(a: argparse.Namespace) -> tuple[list[str], Path]:
    if a.dtype != "fp16":
        sys.exit("ccv lane: the driver is wired for fp16 inputs only")
    bin_ = REPO / "reference_air/ccv/bench_ccv/bench_ccv_attn"
    cmd = [
        str(bin_),
        "--b", "1",
        "--r", str(a.seq),
        "--c", str(a.seq),
        "--hq", str(a.heads),
        "--hk", str(a.heads),
        "--d", str(a.head_dim),
        "--iterations", str(a.iters),
        "--warmup", str(a.warmup),
        "--dispatches", str(a.dispatches),
    ]
    if a.check:
        cmd.append("--check")
    return cmd, bin_


def ccv_build() -> tuple[list[str], Path]:
    return ["bash", "build.sh"], REPO / "reference_air/ccv/bench_ccv"


def mojo_cmd(a: argparse.Namespace) -> tuple[list[str], Path]:
    if a.dtype != "fp16":
        sys.exit("mojo lane: v0 kernel is fp16-in/fp32-out only")
    bin_ = REPO / "bench/build/bench_mojo_metal"
    cmd = [
        str(bin_),
        "--seq", str(a.seq),
        "--head-dim", str(a.head_dim),
        "--heads", str(a.heads),
        "--iters", str(a.iters),
        "--warmup", str(a.warmup),
        "--dispatches", str(a.dispatches),
    ]
    if a.check:
        cmd.append("--check")
    return cmd, bin_


def mojo_build() -> tuple[list[str], Path]:
    (REPO / "bench/build").mkdir(parents=True, exist_ok=True)
    return [
        str(REPO / ".venv/bin/mojo"), "build",
        "bench/bench_mojo_metal.mojo", "-o", "bench/build/bench_mojo_metal",
    ], REPO


IMPLS = {
    "mojo": {"cmd": mojo_cmd, "build": mojo_build},
    "mfa": {"cmd": mfa_cmd, "build": mfa_build},
    "ccv": {"cmd": ccv_cmd, "build": ccv_build},
}


def run_once(name: str, a: argparse.Namespace) -> dict:
    cmd, bin_ = IMPLS[name]["cmd"](a)
    if not bin_.exists():
        sys.exit(f"{name}: binary missing at {bin_} — run with --build first")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"{name} failed:\n{out.stdout}\n{out.stderr}")
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        sys.exit(f"{name}: last stdout line is not JSON:\n{out.stdout}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dtype", default="fp16", choices=list(CHECK_TOL))
    ap.add_argument("--impls", default="mojo,mfa,ccv")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10, help="timed trials per round")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dispatches", type=int, default=5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--build", action="store_true", help="(re)build CLIs first")
    ap.add_argument("--jsonl", type=Path, help="append raw per-run JSON here")
    args = ap.parse_args()
    impls = [i.strip() for i in args.impls.split(",") if i.strip()]
    for i in impls:
        if i not in IMPLS:
            ap.error(f"unknown impl '{i}' (have: {', '.join(IMPLS)})")

    if args.build:
        for i in impls:
            if IMPLS[i]["build"]:
                cmd, cwd = IMPLS[i]["build"]()
                print(f"[build] {i}: {' '.join(cmd)}  ({cwd})", file=sys.stderr)
                subprocess.run(cmd, cwd=cwd, check=True)

    trials: dict[str, list[float]] = {i: [] for i in impls}
    round_medians: dict[str, list[float]] = {i: [] for i in impls}
    check_errs: dict[str, float] = {}
    for r in range(args.rounds):
        for i in impls:  # round-robin interleave
            res = run_once(i, args)
            us = res["gpu_time_us"]
            trials[i].extend(us)
            round_medians[i].append(statistics.median(us))
            if "check_max_error" in res:
                check_errs[i] = max(check_errs.get(i, 0.0), res["check_max_error"])
            if args.jsonl:
                res["round"] = r
                with args.jsonl.open("a") as f:
                    f.write(json.dumps(res) + "\n")
            print(
                f"[round {r}] {i}: median "
                f"{statistics.median(us):.1f} us", file=sys.stderr,
            )

    flops = 4 * args.seq * args.seq * args.head_dim * args.heads
    shape = f"S={args.seq} H={args.heads} D={args.head_dim} {args.dtype}"
    print(f"\n== fwd attention, {shape}, {args.rounds}x{args.iters} trials "
          f"({args.dispatches} dispatches/cb) ==")
    header = f"{'impl':<6} {'min us':>10} {'median us':>11} {'spread':>7} {'GFLOPS':>7}"
    refs = [i for i in impls if i != "mojo"]
    if "mojo" in impls:
        header += "".join(f" {'vs ' + r:>9}" for r in refs)
    print(header)
    failed = []
    for i in impls:
        mn = min(trials[i])
        md = statistics.median(trials[i])
        rm = round_medians[i]
        spread = (max(rm) - min(rm)) / statistics.median(rm) * 100
        row = (
            f"{i:<6} {mn:>10.1f} {md:>11.1f} {spread:>6.1f}% "
            f"{flops / (mn * 1000):>7.0f}"
        )
        if "mojo" in impls:
            if i == "mojo":
                row += "".join(
                    f" {min(trials['mojo']) / min(trials[r]):>8.3f}x" for r in refs
                )
            else:
                row += " " * 10 * len(refs)
        print(row)
        if args.check:
            tol = CHECK_TOL[args.dtype]
            err = check_errs.get(i)
            status = (
                "no check reported" if err is None
                else f"max|err| {err:.2e} vs tol {tol:.0e}"
            )
            ok = err is not None and err <= tol
            print(f"       check: {status} {'OK' if ok else 'FAIL'}")
            if not ok:
                failed.append(i)
    if failed:
        sys.exit(f"correctness FAILED for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
