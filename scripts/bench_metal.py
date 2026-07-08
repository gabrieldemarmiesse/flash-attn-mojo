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
(4*S*S*D) and mojo-vs-reference ratios (<1 means mojo is faster,
same convention as the FA4 race).

Single dispatch per command buffer flaps 5-50x on M4 (GPU power
states) — keep --dispatches >= 5.

The structured entry point is ``bench_shape(...)`` (returns a dict of
per-impl aggregated stats); ``scripts/master_bench.py`` calls it
to build the rich comparison / roofline / ratchet tables. Run directly
for the standalone human table.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent

# Per-dtype max|err| gates vs each CLI's built-in fp32 reference check.
# (MFA measures ~1e-5 at fp16 inputs/intermediates; gates are loose
# enough for run-to-run input variation, tight enough to catch a
# broken mask/softmax, which shows up at >=1e-2.)
CHECK_TOL = {"fp16": 1e-3, "fp32": 5e-5, "bf16": 1e-2}


class BenchError(RuntimeError):
    """A lane failed to build/run/parse — surfaced so a coordinator can
    treat it as a gate failure instead of exiting the whole process."""


def mfa_cmd(a) -> tuple[list[str], Path]:
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


def ccv_cmd(a) -> tuple[list[str], Path]:
    if a.dtype != "fp16":
        raise BenchError("ccv lane: the driver is wired for fp16 inputs only")
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


def mojo_cmd(a) -> tuple[list[str], Path]:
    if a.dtype != "fp16":
        raise BenchError("mojo lane: v0 kernel is fp16-in/fp32-out only")
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


def run_once(name: str, a) -> dict:
    cmd, bin_ = IMPLS[name]["cmd"](a)
    if not bin_.exists():
        raise BenchError(f"{name}: binary missing at {bin_} — build it first")
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise BenchError(f"{name} failed:\n{out.stdout}\n{out.stderr}")
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise BenchError(f"{name}: last stdout line is not JSON:\n{out.stdout}")


def build_impls(impls, *, log=lambda m: None) -> None:
    for i in impls:
        if IMPLS[i]["build"]:
            cmd, cwd = IMPLS[i]["build"]()
            log(f"[build] {i}: {' '.join(cmd)}  ({cwd})")
            subprocess.run(cmd, cwd=cwd, check=True)


def bench_shape(
    impls,
    *,
    seq,
    heads,
    head_dim,
    dtype="fp16",
    rounds=3,
    iters=10,
    warmup=3,
    dispatches=5,
    check=False,
    jsonl: Path | None = None,
    progress=None,
) -> dict:
    """Run the interleaved 3-way bench for one shape.

    Returns ``{impl: {trials, round_medians, min, median, spread_pct,
    check_err}}`` — the aggregated per-impl stats a coordinator (or the
    standalone table below) turns into ratios / roofline / ratchet.
    ``progress`` is an optional ``callable(str)`` for per-round lines.
    """
    a = SimpleNamespace(
        seq=seq, heads=heads, head_dim=head_dim, dtype=dtype,
        iters=iters, warmup=warmup, dispatches=dispatches, check=check,
    )
    trials: dict[str, list[float]] = {i: [] for i in impls}
    round_medians: dict[str, list[float]] = {i: [] for i in impls}
    check_errs: dict[str, float] = {}
    for r in range(rounds):
        for i in impls:  # round-robin interleave
            res = run_once(i, a)
            us = res["gpu_time_us"]
            trials[i].extend(us)
            round_medians[i].append(statistics.median(us))
            if "check_max_error" in res:
                check_errs[i] = max(check_errs.get(i, 0.0), res["check_max_error"])
            if jsonl:
                res["round"] = r
                res["shape"] = {"seq": seq, "heads": heads, "head_dim": head_dim,
                                "dtype": dtype}
                with jsonl.open("a") as f:
                    f.write(json.dumps(res) + "\n")
            if progress:
                progress(f"[round {r}] {i}: median {statistics.median(us):.1f} us")

    out: dict[str, dict] = {}
    for i in impls:
        rm = round_medians[i]
        spread = (max(rm) - min(rm)) / statistics.median(rm) * 100 if rm else float("nan")
        out[i] = {
            "trials": trials[i],
            "round_medians": rm,
            "min": min(trials[i]),
            "median": statistics.median(trials[i]),
            "spread_pct": spread,
            "check_err": check_errs.get(i),
        }
    return out


def _print_table(results: dict, impls, seq, heads, head_dim, dtype,
                 rounds, iters, dispatches, check) -> list[str]:
    flops = 4 * seq * seq * head_dim * heads
    shape = f"S={seq} H={heads} D={head_dim} {dtype}"
    print(f"\n== fwd attention, {shape}, {rounds}x{iters} trials "
          f"({dispatches} dispatches/cb) ==")
    header = f"{'impl':<6} {'min us':>10} {'median us':>11} {'spread':>7} {'GFLOPS':>7}"
    refs = [i for i in impls if i != "mojo"]
    if "mojo" in impls:
        header += "".join(f" {'vs ' + r:>9}" for r in refs)
    print(header)
    failed = []
    for i in impls:
        st = results[i]
        row = (
            f"{i:<6} {st['min']:>10.1f} {st['median']:>11.1f} {st['spread_pct']:>6.1f}% "
            f"{flops / (st['min'] * 1000):>7.0f}"
        )
        if "mojo" in impls:
            if i == "mojo":
                row += "".join(
                    f" {results['mojo']['min'] / results[r]['min']:>8.3f}x" for r in refs
                )
            else:
                row += " " * 10 * len(refs)
        print(row)
        if check:
            tol = CHECK_TOL[dtype]
            err = st["check_err"]
            status = (
                "no check reported" if err is None
                else f"max|err| {err:.2e} vs tol {tol:.0e}"
            )
            ok = err is not None and err <= tol
            print(f"       check: {status} {'OK' if ok else 'FAIL'}")
            if not ok:
                failed.append(i)
    return failed


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

    try:
        if args.build:
            build_impls(impls, log=lambda m: print(m, file=sys.stderr))
        results = bench_shape(
            impls,
            seq=args.seq, heads=args.heads, head_dim=args.head_dim, dtype=args.dtype,
            rounds=args.rounds, iters=args.iters, warmup=args.warmup,
            dispatches=args.dispatches, check=args.check, jsonl=args.jsonl,
            progress=lambda m: print(m, file=sys.stderr),
        )
    except BenchError as e:
        sys.exit(str(e))

    failed = _print_table(results, impls, args.seq, args.heads, args.head_dim,
                          args.dtype, args.rounds, args.iters, args.dispatches, args.check)
    if failed:
        sys.exit(f"correctness FAILED for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
