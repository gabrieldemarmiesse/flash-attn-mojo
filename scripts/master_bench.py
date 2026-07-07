#!/usr/bin/env python3
"""master_bench.py — the one autonomous perf gate for the FA4-vs-Mojo race.

A structured, high-information coordinator around the flash-attn-mojo kernels.
The headline target is ``flash_attn_func(q, k, v)`` (the dense fwd), but every
axis the harness already understands — bwd, causal, GQA, varlen, window,
softcap, fp16, hdim64 — rides through the same flags, so this one script
covers the whole envelope from a single entry point.

It is a stdlib-only coordinator: the real work happens in subprocesses under
the project venv (``uv run python scripts/bench_fa4.py …`` for timing +
correctness, ``ptxas`` / ``scripts/ptx_stats.py`` for the asm phases). This
script detects the GPU, locks the clocks, sequences the phases, parses their
output into rich tables, and gates on the results.

Phases (each announces itself; unavailable tooling skips cleanly):

    a. lock clocks              nvidia-smi (HARD GATE — an unlocked H100 drifts
                                ±4-5% and fakes wins/losses; --no-lock opts out)
    b. clear JIT cache + recompile + correctness suite (vs fp32 refs + FA4)
    c. kernel-time bench vs FA4 — per-shape ratio table (mojo us | fa4 us |
       ratio | run-to-run spread | verdict) + a compute (tensor-core) roofline
       verdict per shape (achieved TFLOP/s, % of peak, regime, actionable hint)
    d. ncu deep profiler — per-metric summary for BOTH kernels side by side
       (SM/DRAM/tensor-pipe throughput, occupancy, waves), mean + spread
    e. dump the mojo kernel's PTX -> ptx/
    f. PTX instruction-mix histogram: mojo vs the committed FA4 reference PTX
    g. ptxas -v spill / regalloc canary (HARD GATE — both kernels must stay 0)
    h. independent end-to-end wall-clock run (CUDA-event timed, launch + sync
       overhead included) — catches launch-bound regressions CUPTI can't see

The kernel-time bench (c) is the true perf GATE: a mojo shape slower than FA4
by more than the measured run-to-run spread fails the run (non-zero exit),
unless --no-gate. Everything else is informational unless it is a correctness
or spill failure (both hard gates).

Usage:
    scripts/master_bench.py                       # quick tier, canonical fwd
    scripts/master_bench.py --kind bwd            # backward kernels
    scripts/master_bench.py --causal --hkv 4      # causal GQA fwd
    scripts/master_bench.py --varlen              # packed varlen
    scripts/master_bench.py --full                # multi-shape sweep
    scripts/master_bench.py --no-ncu --no-asm     # timing only (fast loop)
    scripts/master_bench.py --no-lock --no-clean  # dev loop, keep JIT cache
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parent.parent
BENCH = str(REPO / "scripts" / "bench_fa4.py")
# Absolute path: the ncu phase re-invokes the bench under `sudo -E`, whose
# reset PATH would not otherwise find a bare `uv`.
UV = shutil.which("uv") or "uv"

# --------------------------------------------------------------------------
# Shape sweeps. The headline is the canonical FA4-race shape; --full adds a
# handful of neighbours that stress different wave counts / hidden dims.
# --------------------------------------------------------------------------
CANON = "2,8192,16,128"
FULL_SHAPES = [
    "2,8192,16,128",  # canonical (many-wave, compute-bound)
    "1,4096,16,128",  # half batch, half seq
    "4,4096,16,128",  # more batch
    "1,2048,32,128",  # short seq, more heads
    "2,8192,32,64",   # hdim64
]

# bf16/fp16 dense tensor-core peak (TFLOP/s) by device-name substring, for the
# compute roofline. Attention is matmul-bound, so the binding roofline is the
# tensor core, not DRAM. Longest match wins; FLASH_ATTN_MOJO_PEAK_TFLOPS
# overrides; an unknown device omits the %-of-peak column.
_PEAK_TFLOPS = {
    "H100 NVL": 835.0,
    "H100 PCIe": 756.0,
    "H200": 989.0,
    "H100": 989.0,  # SXM
    "A100": 312.0,
    "L40S": 362.0,
    "RTX 5090": 838.0,
    "RTX 4090": 330.0,
}

# ncu metrics — all standard on sm_90; SM/DRAM/tensor-pipe throughput,
# occupancy, wave count. Kept short so a replay pass stays cheap.
_NCU_METRICS = [
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__waves_per_multiprocessor",
]
_NCU_LAUNCH_SKIP = 3
_NCU_LAUNCH_COUNT = 5

_BOLD, _RST, _YEL, _RED, _GRN = (
    "\033[1m", "\033[0m", "\033[33m", "\033[31m", "\033[32m",
)
VERBOSE = False


def section(msg: str) -> None:
    print(f"\n{_BOLD}==== {msg} ===={_RST}", flush=True)


def warn(msg: str) -> None:
    print(f"{_YEL}[warn]{_RST} {msg}", file=sys.stderr, flush=True)


def skip(step: str, reason: str) -> None:
    section(step)
    print(f"skipped — {reason}")


class Gate:
    """Accumulates gate failures; the process exit code reflects it."""

    failed = False

    @classmethod
    def fail(cls, msg: str) -> None:
        cls.failed = True
        print(f"{_RED}[FAIL]{_RST} {msg}", file=sys.stderr, flush=True)


def run(cmd, *, env=None, capture=False):
    if VERBOSE:
        print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


# --------------------------------------------------------------------------
# Device probing (stdlib-only — no torch import in the coordinator).
# --------------------------------------------------------------------------


def nvidia_smi(query: str) -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    out = r.stdout.strip().splitlines()
    return out[0].strip() if out else ""


def peak_tflops(gpu: str) -> float | None:
    override = os.environ.get("FLASH_ATTN_MOJO_PEAK_TFLOPS")
    if override:
        return float(override)
    for name in sorted(_PEAK_TFLOPS, key=len, reverse=True):
        if name in gpu:
            return _PEAK_TFLOPS[name]
    return None


# --------------------------------------------------------------------------
# (a) Lock clocks — a hard gate (an unlocked H100 makes runs incomparable).
# --------------------------------------------------------------------------


def _fatal_lock_failure(reason: str) -> NoReturn:
    print(f"{_RED}[FAIL]{_RST} GPU clock lock unavailable — refusing to run unlocked.")
    print(f"{_RED}[FAIL]{_RST} ({reason})")
    print(f"{_RED}[FAIL]{_RST} pass --no-lock to run unlocked anyway (dev mode).")
    raise SystemExit(1)


def lock_clocks(enabled: bool) -> tuple[str, bool]:
    section("(a) lock clocks")
    if not enabled:
        warn("--no-lock: clocks NOT locked (dev mode — numbers are not comparable).")
        return "unlocked", False

    def sudo(args) -> bool:
        return (
            subprocess.run(
                ["sudo", "-n", "nvidia-smi", *args], capture_output=True, text=True
            ).returncode
            == 0
        )

    max_sm = nvidia_smi("clocks.max.sm")
    if max_sm and sudo(["-pm", "1"]) and sudo([f"--lock-gpu-clocks={max_sm},{max_sm}"]):
        max_mem = nvidia_smi("clocks.max.mem")
        if max_mem:
            sudo([f"--lock-memory-clocks={max_mem},{max_mem}"])  # best-effort
        print(f"locked SM clock to {max_sm} MHz (resets on exit)")
        return max_sm, True
    _fatal_lock_failure(
        "passwordless `sudo -n nvidia-smi -pm 1 --lock-gpu-clocks=…` unavailable"
    )


def unlock_clocks() -> None:
    for args in (["--reset-gpu-clocks"], ["--reset-memory-clocks"]):
        subprocess.run(["sudo", "-n", "nvidia-smi", *args], capture_output=True)


# --------------------------------------------------------------------------
# Variant plumbing: CLI flags -> bench flags + PTX/ncu-filter names.
# --------------------------------------------------------------------------


def variant(args) -> dict:
    """Everything the phases need to know about the selected variant."""
    hdim = args.shape.split(",")[-1] if not args.varlen else "128"
    csuf, msuf, cflag = "noncausal", "", []
    if args.causal:
        csuf, msuf = "causal", "_causal"
        cflag.append("--causal")
    if args.hkv:
        csuf, msuf = f"{csuf}_gqa", f"{msuf}_gqa"
        cflag += ["--hkv", str(args.hkv)]
    if args.varlen:
        csuf, msuf = f"{csuf}_varlen", f"{msuf}_varlen"
        cflag.append("--varlen")
    if args.window:
        csuf, msuf = f"{csuf}_window", f"{msuf}_win"
        cflag += ["--window", str(args.window)]
    if args.softcap:
        csuf, msuf = f"{csuf}_softcap", f"{msuf}_scap"
        cflag += ["--softcap", str(args.softcap)]
    if hdim != "128":
        msuf = f"{msuf}_hd{hdim}"
    if args.dtype != "bf16":
        cflag += ["--dtype", args.dtype]
    if args.kind == "fwd":
        fa4_ptx = REPO / "reference_ptx" / f"fa4_fwd_sm90_bf16_hdim{hdim}_{csuf}.ptx"
        mojo_ptx = REPO / "ptx" / f"mojo_fwd_fa4{msuf}.ptx"
        fa4_filter, mojo_filter = "FlashAttentionForwardSm90", "fwd_fa4_kernel"
    else:
        fa4_ptx = REPO / "reference_ptx" / f"fa4_bwd_sm90_bf16_hdim{hdim}_{csuf}.ptx"
        mojo_ptx = REPO / "ptx" / f"mojo_bwd_fa4{msuf}.ptx"
        fa4_filter, mojo_filter = "FlashAttentionBackwardSm90", "bwd_main_kernel"
    return {
        "cflag": cflag,
        "fa4_ptx": fa4_ptx,
        "mojo_ptx": mojo_ptx,
        "fa4_filter": fa4_filter,
        "mojo_filter": mojo_filter,
    }


def bench_cmd(args, var, impl, shape, *, extra=()):
    return [
        UV, "run", "python", BENCH,
        "--impl", impl, "--kind", args.kind, "--shape", shape,
        "--iters", str(args.iters), *var["cflag"], *extra,
    ]


_RESULT_RE = re.compile(r"us=([0-9.]+)\s+tflops=([0-9.]+)")


def parse_results(text: str) -> list[tuple[float, float]]:
    out = []
    for line in text.splitlines():
        if line.startswith("RESULT"):
            m = _RESULT_RE.search(line)
            if m:
                out.append((float(m.group(1)), float(m.group(2))))
    return out


# --------------------------------------------------------------------------
# (b) Recompile (clear OUR JIT cache) + correctness.
# --------------------------------------------------------------------------


def correctness(args, var, clean: bool) -> None:
    section("(b) clear JIT cache + recompile + correctness")
    if clean:
        cache = Path("~/.cache/flash_attn_mojo").expanduser()
        if VERBOSE:
            print(f"clearing JIT cache {cache} (mojo compiler cache untouched)")
        shutil.rmtree(cache, ignore_errors=True)
    else:
        print("(--no-clean: keeping JIT cache)")
    cmd = bench_cmd(args, var, "mojo", args.shape, extra=["--check-only"])
    if run(cmd).returncode != 0:
        Gate.fail("correctness failed (see pytest output above)")
    else:
        print(f"{_GRN}correctness OK{_RST}")


# --------------------------------------------------------------------------
# (c) Kernel-time bench vs FA4 + compute roofline.
# --------------------------------------------------------------------------


def _spread_pct(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    lo, hi = min(vals), max(vals)
    return (hi - lo) / lo * 100.0 if lo else 0.0


def _roofline(tflops: float, peak: float | None) -> tuple[str, str, float | None]:
    """(regime, hint, pct_peak) for a compute-bound attention kernel."""
    if not peak:
        return ("unknown", "set FLASH_ATTN_MOJO_PEAK_TFLOPS for a %-of-peak", None)
    pct = 100.0 * tflops / peak
    if pct >= 80:
        return (
            "compute-bound (near-peak)",
            "at the tensor-core roofline — the FA4 parity race is the only lever",
            pct,
        )
    if pct >= 50:
        return (
            "compute-bound",
            "healthy TC utilization — chase the FA4 gap via the PTX/SASS op-mix",
            pct,
        )
    if pct >= 25:
        return (
            "partial",
            "mid utilization — check occupancy / waves / pipeline stalls in (d)",
            pct,
        )
    return (
        "latency/occupancy-bound",
        "small wave — grow B*S or inspect scheduler stalls (ncu source page)",
        pct,
    )


def bench(args, var, shapes, peak, gate: bool) -> list[dict]:
    section(f"(c) kernel-time bench vs FA4  (min of {args.runs} runs/impl, "
            "locked clock)")
    rows = []
    for shape in shapes:
        row = {"shape": shape}
        for impl in ("fa4", "mojo"):
            r = run(bench_cmd(args, var, impl, shape, extra=["--runs", str(args.runs)]),
                    capture=True)
            if r.stderr and VERBOSE:
                sys.stderr.write(r.stderr)
            res = parse_results(r.stdout or "")
            if r.returncode != 0 or not res:
                Gate.fail(f"{impl} bench failed on {shape}")
                if r.stderr:
                    sys.stderr.write(r.stderr[-800:])
                row[impl] = None
                continue
            us = [u for u, _ in res]
            tf = [t for _, t in res]
            row[impl] = {
                "us": min(us),
                "spread": _spread_pct(us),
                "tflops": max(tf),
            }
        rows.append(row)

    hdr = (
        f"  {'shape':>16} | {'mojo us':>9} | {'fa4 us':>9} | {'ratio':>7} | "
        f"{'spread':>7} | {'mojo TF/s':>9} | verdict"
    )
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    worst = 0.0
    for row in rows:
        mojo, fa4 = row.get("mojo"), row.get("fa4")
        if not mojo or not fa4:
            print(f"  {row['shape']:>16} | {'--':>9} | {'--':>9} | "
                  f"{'--':>7} | {'--':>7} | {'--':>9} | NO-DATA")
            continue
        ratio = mojo["us"] / fa4["us"]
        spread = max(mojo["spread"], fa4["spread"])
        gap = abs(ratio - 1) * 100
        worst = max(worst, ratio)
        if gap <= 3:
            verdict = "parity"
        elif gap < spread:
            verdict = "NOISE"
        elif ratio < 1:
            verdict = f"{_GRN}FASTER{_RST}"
        else:
            verdict = f"{_RED}SLOWER{_RST}"
            if gate:
                Gate.fail(
                    f"perf regression on {row['shape']}: {ratio:.3f}x "
                    f"(gap {gap:.1f}% > spread {spread:.1f}%)"
                )
        print(
            f"  {row['shape']:>16} | {mojo['us']:9.1f} | {fa4['us']:9.1f} | "
            f"{ratio:6.3f}x | {spread:6.1f}% | {mojo['tflops']:9.1f} | {verdict}"
        )
    print("  " + "-" * (len(hdr) - 2))
    if worst:
        tail = "(<=1.03x everywhere)" if worst <= 1.03 else "(a shape exceeds 1.03x)"
        print(f"  worst mojo/fa4 ratio: {worst:.3f}x  {tail}")

    # Compute (tensor-core) roofline per shape.
    print(f"\n  compute roofline"
          + (f" (peak = {peak:.0f} TFLOP/s bf16 tensor)" if peak else " (peak unknown)"))
    rhdr = f"  {'shape':>16} | {'TFLOP/s':>9} | {'%peak':>6} | regime"
    print(rhdr)
    print("  " + "-" * (len(rhdr) - 2))
    hints = []
    for row in rows:
        mojo = row.get("mojo")
        if not mojo:
            continue
        regime, hint, pct = _roofline(mojo["tflops"], peak)
        row["roofline"] = {"regime": regime, "hint": hint, "pct_peak": pct}
        pct_s = f"{pct:.0f}%" if pct is not None else "?"
        print(f"  {row['shape']:>16} | {mojo['tflops']:9.1f} | {pct_s:>6} | {regime}")
        if regime not in ("compute-bound (near-peak)",):
            hints.append((row["shape"], hint))
    for shape, hint in hints:
        print(f"    {shape}: {hint}")
    return rows


# --------------------------------------------------------------------------
# (d) ncu deep profiler — per-metric summary for both kernels side by side.
# --------------------------------------------------------------------------


def _ncu_cmd() -> list[str] | None:
    # Resolve to absolute paths: the ncu phase may re-invoke under `sudo`,
    # which resets PATH and would not find a bare `ncu`/`pixi`.
    ncu = shutil.which("ncu")
    if ncu:
        return [ncu]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "exec", "--spec", "nsight-compute=2024.3.2", "--", "ncu"]
    return None


def _needs_sudo() -> bool:
    try:
        params = Path("/proc/driver/nvidia/params").read_text()
    except OSError:
        return False
    return any(
        ln.strip() == "RmProfilingAdminOnly: 1" for ln in params.splitlines()
    )


def _parse_ncu_csv(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines)
         if '"Metric Name"' in ln and '"Metric Value"' in ln),
        None,
    )
    if start is None:
        return {}
    data: dict[str, dict] = {}
    for row in csv.DictReader(lines[start:]):
        name = (row.get("Metric Name") or "").strip()
        raw = (row.get("Metric Value") or "").strip()
        if not name or not raw:
            continue
        try:
            num = float(raw.replace(",", ""))
        except ValueError:
            continue
        d = data.setdefault(name, {"unit": (row.get("Metric Unit") or "").strip(),
                                   "vals": []})
        d["vals"].append(num)
    return data


def _ncu_capture(ncu, sudo, args, var, impl, filt) -> dict[str, dict]:
    cmd = [
        *sudo, *ncu,
        "--target-processes", "all",
        "--kernel-name", f"regex:{filt}",
        "--launch-skip", str(_NCU_LAUNCH_SKIP),
        "--launch-count", str(_NCU_LAUNCH_COUNT),
        "--metrics", ",".join(_NCU_METRICS),
        "--csv",
        UV, "run", "python", BENCH,
        "--impl", impl, "--kind", args.kind, "--shape", args.shape,
        "--profile", "--iters", str(_NCU_LAUNCH_SKIP + _NCU_LAUNCH_COUNT + 2),
        "--warmup", "3", *var["cflag"],
    ]
    if VERBOSE:
        print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE)
    if proc.returncode != 0:
        warn(f"ncu capture failed for {impl} (perf-counter perms?)")
        return {}
    return _parse_ncu_csv(proc.stdout or "")


def profiler(args, var, enabled: bool) -> None:
    if not enabled:
        skip("(d) ncu deep profiler", "--no-ncu")
        return
    section("(d) ncu deep profiler — fa4 vs mojo")
    ncu = _ncu_cmd()
    if ncu is None:
        warn("ncu not found and pixi unavailable — skipping deep profiling.")
        return
    sudo = []
    if _needs_sudo():
        if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
            sudo = ["sudo", "-E"]
        else:
            warn("RmProfilingAdminOnly=1 and no passwordless sudo — skipping ncu.")
            return
    fa4 = _ncu_capture(ncu, sudo, args, var, "fa4", var["fa4_filter"])
    mojo = _ncu_capture(ncu, sudo, args, var, "mojo", var["mojo_filter"])
    if not fa4 and not mojo:
        return

    def cell(d: dict, name: str) -> str:
        m = d.get(name)
        if not m or not m["vals"]:
            return "-"
        vals = m["vals"]
        mean = sum(vals) / len(vals)
        return f"{mean:.2f} (±{_spread_pct(vals):.0f}%)"

    hdr = f"    {'metric':<52} | {'unit':>7} | {'fa4':>16} | {'mojo':>16}"
    print(f"\n{hdr}")
    print("    " + "-" * (len(hdr) - 4))
    for name in _NCU_METRICS:
        unit = (fa4.get(name) or mojo.get(name) or {}).get("unit", "")
        print(f"    {name:<52} | {unit:>7} | {cell(fa4, name):>16} | "
              f"{cell(mojo, name):>16}")
    n = max(
        [len(m["vals"]) for m in list(fa4.values()) + list(mojo.values())],
        default=0,
    )
    print(f"    (mean ± run-to-run spread over {n} profiled launches per kernel)")


# --------------------------------------------------------------------------
# (e) dump PTX  (f) instruction-mix histogram  (g) spill canary.
# --------------------------------------------------------------------------


def _find_ptxas() -> str | None:
    """ptxas is not on PATH here (mojo compiles PTX in-process via
    libNVPTX.so), but the torch/triton wheels in the venv ship one — good
    enough for the spill/regalloc canary (version is not a perf variable)."""
    direct = shutil.which("ptxas")
    if direct:
        return direct
    for pat in (
        ".venv/lib/python*/site-packages/torch/bin/ptxas",
        ".venv/lib/python*/site-packages/triton/backends/nvidia/bin/ptxas",
    ):
        hits = sorted(REPO.glob(pat))
        if hits:
            return str(hits[0])
    return None


def assembly(args, var, enabled: bool, refresh_fa4: bool) -> None:
    if not enabled:
        skip("(e/f/g) PTX dump + histogram + spill canary", "--no-asm")
        return
    mojo_ptx = var["mojo_ptx"]
    mojo_ptx.parent.mkdir(parents=True, exist_ok=True)

    section("(e) dump mojo PTX -> ptx/")
    env = {**os.environ, "MOJO_DUMP_PTX": str(mojo_ptx)}
    cmd = bench_cmd(args, var, "mojo", args.shape,
                    extra=["--iters", "1", "--warmup", "0"])
    if run(cmd, env=env, capture=not VERBOSE).returncode != 0 or not mojo_ptx.exists():
        Gate.fail("PTX dump failed")
        return
    print(f"mojo PTX: {mojo_ptx.relative_to(REPO)}")

    section("(g) ptxas -v spill / regalloc canary")
    ptxas = _find_ptxas()
    if not ptxas:
        warn("ptxas not found (PATH or venv) — skipping spill canary")
    else:
        r = run([ptxas, "-arch=sm_90a", "-v", str(mojo_ptx)], capture=True)
        blob = (r.stderr or "") + (r.stdout or "")
        spill = 0
        for m in re.finditer(r"(\d+)\s+bytes spill stores", blob):
            spill += int(m.group(1))
        regs = re.search(r"Used (\d+) registers", blob)
        smem = re.search(r"(\d+) bytes smem", blob)
        print(f"  registers: {regs.group(1) if regs else '?'} | "
              f"smem: {smem.group(1) if smem else '?'} B | "
              f"spill stores: {spill} B")
        if spill > 0:
            Gate.fail(f"spill canary: {spill} bytes spilled (must be 0)")
        else:
            print(f"  {_GRN}no spills{_RST}")

    section("(f) PTX instruction-mix histogram: fa4 (reference) vs mojo")
    fa4_ptx = var["fa4_ptx"]
    if refresh_fa4:
        warn("--refresh-fa4-ptx: regenerate reference PTX via "
             "master_bench.sh --refresh-fa4-ptx (not automated here)")
    if not fa4_ptx.exists():
        warn(f"no FA4 reference PTX at {fa4_ptx.relative_to(REPO)} — "
             "showing mojo op-mix only")
        run([UV, "run", "python", str(REPO / "scripts" / "ptx_stats.py"),
             str(mojo_ptx)])
    else:
        run([UV, "run", "python", str(REPO / "scripts" / "ptx_stats.py"),
             str(fa4_ptx), str(mojo_ptx)])


# --------------------------------------------------------------------------
# (h) Independent wall-clock run.
# --------------------------------------------------------------------------


def walltime(args, var) -> None:
    section("(h) end-to-end wall-clock (CUDA-event timed, launch+sync included)")
    rows = []
    for impl in ("fa4", "mojo"):
        r = run(bench_cmd(args, var, impl, args.shape,
                          extra=["--walltime", "--runs", str(args.runs)]),
                capture=True)
        res = parse_results(r.stdout or "")
        if r.returncode != 0 or not res:
            Gate.fail(f"walltime {impl} run failed")
            if r.stderr:
                sys.stderr.write(r.stderr[-500:])
            rows.append(None)
            continue
        us = [u for u, _ in res]
        rows.append({"us": min(us), "spread": _spread_pct(us)})
    fa4, mojo = rows
    if fa4 and mojo:
        ratio = mojo["us"] / fa4["us"]
        print(f"  fa4:  {fa4['us']:8.1f} us  (±{fa4['spread']:.1f}%)")
        print(f"  mojo: {mojo['us']:8.1f} us  (±{mojo['spread']:.1f}%)")
        note = "mojo FASTER" if ratio < 1 else "mojo slower"
        print(f"  wall-clock ratio: {ratio:.3f}x  ({note})")
        # A big kernel-time/walltime gap => launch-bound; surface it.
        print("  (compare vs the (c) kernel-time ratio: a wall-clock-only "
              "regression is launch/dispatch overhead, not the kernel)")


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--kind", choices=("fwd", "bwd"), default="fwd")
    p.add_argument("--shape", default=CANON, help=f"B,S,H,D (default {CANON})")
    p.add_argument("--full", action="store_true", help="multi-shape sweep")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--hkv", type=int, default=0, help="KV heads (GQA); 0=MHA")
    p.add_argument("--varlen", action="store_true")
    p.add_argument("--window", type=int, default=0, help="sliding-window left")
    p.add_argument("--softcap", type=float, default=0.0)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--runs", type=int, default=3, help="timed runs/impl for spread")
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--no-clean", action="store_true", help="keep the JIT cache")
    p.add_argument("--no-check", action="store_true", help="skip correctness")
    p.add_argument("--no-ncu", action="store_true")
    p.add_argument("--no-asm", action="store_true")
    p.add_argument("--no-walltime", action="store_true")
    p.add_argument("--no-gate", action="store_true",
                   help="report a SLOWER shape but don't fail the run")
    p.add_argument("--refresh-fa4-ptx", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    global VERBOSE
    VERBOSE = args.verbose
    os.chdir(REPO)

    if not (shutil.which("nvidia-smi") and nvidia_smi("name")):
        print(f"{_RED}[FAIL]{_RST} no NVIDIA GPU detected — this bench is CUDA-only.")
        return 1
    gpu = nvidia_smi("name")
    peak = peak_tflops(gpu)
    var = variant(args)
    shapes = FULL_SHAPES if args.full else [args.shape]
    axes = [args.kind]
    if args.causal:
        axes.append("causal")
    if args.hkv:
        axes.append(f"gqa(hkv={args.hkv})")
    if args.varlen:
        axes.append("varlen")
    if args.window:
        axes.append(f"window={args.window}")
    if args.softcap:
        axes.append(f"softcap={args.softcap:g}")
    axes.append(args.dtype)
    peak_note = f"  peak={peak:.0f} TFLOP/s" if peak else "  (peak unknown)"
    print(f"master_bench: device='{gpu}'{peak_note}")
    print(f"variant: {' + '.join(axes)}   shapes: {', '.join(shapes)}")

    clock, locked = lock_clocks(not args.no_lock)
    if locked:
        def _on_signal(signum, _frame):
            unlock_clocks()
            os._exit(128 + signum)

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGHUP, _on_signal)

    rows: list[dict] = []
    try:
        if args.no_check:
            skip("(b) correctness", "--no-check")
        else:
            correctness(args, var, not args.no_clean)
        rows = bench(args, var, shapes, peak, gate=not args.no_gate)
        profiler(args, var, not args.no_ncu)
        assembly(args, var, not args.no_asm, args.refresh_fa4_ptx)
        if not args.no_walltime:
            walltime(args, var)
        else:
            skip("(h) wall-clock", "--no-walltime")
    finally:
        if locked:
            unlock_clocks()

    section("summary")
    agent_rows = []
    for row in rows:
        mojo, fa4 = row.get("mojo"), row.get("fa4")
        rl = row.get("roofline") or {}
        agent_rows.append({
            "shape": row["shape"],
            "mojo_us": round(mojo["us"], 2) if mojo else None,
            "fa4_us": round(fa4["us"], 2) if fa4 else None,
            "ratio": round(mojo["us"] / fa4["us"], 4) if mojo and fa4 else None,
            "mojo_tflops": round(mojo["tflops"], 1) if mojo else None,
            "pct_peak": rl.get("pct_peak") and round(rl["pct_peak"], 1),
            "regime": rl.get("regime"),
        })
    summary = {
        "device": gpu,
        "peak_tflops": peak,
        "clock": clock,
        "variant": " + ".join(axes),
        "gate": "fail" if Gate.failed else "pass",
        "shapes": agent_rows,
    }
    print("===AGENT-SUMMARY===")
    print(json.dumps(summary))
    print("===END-AGENT-SUMMARY===")
    if Gate.failed:
        print(f"{_RED}ISSUES — one or more gates failed above.{_RST}")
        return 1
    print(f"{_GRN}PASS — all gates green (kind={args.kind} "
          f"variant={' + '.join(axes)}).{_RST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
