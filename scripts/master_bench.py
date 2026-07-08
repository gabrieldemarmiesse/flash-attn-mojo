#!/usr/bin/env python3
"""master_bench.py — the ONE autonomous perf gate for the flash-attn-mojo race.

Run this after any kernel edit and read the summary. It is the single entry
point an agent uses to see how to improve performance — it **auto-detects the
GPU backend** and runs that backend's structured, high-information phases:

    * NVIDIA / CUDA  — mojo vs Tri Dao's FlashAttention-4 (`flash_attn.cute`).
                       Locks clocks (hard gate), clears the JIT cache + runs the
                       fp32-reference correctness suite, benches a per-shape
                       ratio table + tensor-core roofline, captures ncu metrics
                       for both kernels, dumps the mojo PTX, diffs its op-mix vs
                       the committed FA4 reference, runs the ptxas spill canary
                       (hard gate), and does an independent wall-clock run.
    * Apple / Metal  — mojo vs metal-flash-attention (MFA) AND ccv, taking the
                       per-shape best of {MFA, ccv} as the empirical ceiling.
                       Induced-max clock template, 3-way interleaved bench, then
                       (default on) an xctrace pass that recovers the mojo lane's
                       TRUE per-dispatch GPU kernel time — the refs self-report
                       command-buffer GPU time, so this makes the comparison
                       kernel-to-kernel (mojo's wall-clock lane eats ~160 us/
                       dispatch of enqueue overhead the refs don't). Plus a
                       compute roofline, an AIR op-mix diff, and a per-shape
                       best-time regression ratchet. Disable with
                       --no-kernel-time.
    * AMD / ROCm     — mojo v0 vs CK-flash (Tri Dao's `flash_attn`, Composable
                       Kernel backend — the only AMD reference). Builds the mojo
                       kernel, benches vs CK (kernel-only via rocprofv3), dumps
                       the AMDGCN ISA + op-mix, and profiles under rocprofv3.
    * CPU            — `--device cpu`. There is NO GPU kernel on CPU (the public
                       op falls through to the pure-PyTorch reference), so this
                       is a correctness smoke-check (tests/test_api.py) plus a
                       naive reference wall-clock — NOT a perf race.

Every backend ends with a machine-readable `===AGENT-SUMMARY===` JSON block and
a non-zero exit on any gate failure.

Backend selection:
    auto (default) — darwin => metal; else nvidia-smi => cuda; else a torch
                     probe => rocm (HIP) / cpu. Override with
                     `--device {cuda,rocm,metal,cpu}`.

Usage:
    scripts/master_bench.py                       # auto-detect, canonical fwd
    scripts/master_bench.py --device cpu          # CPU reference smoke-check
    scripts/master_bench.py --kind bwd            # [cuda] backward kernels
    scripts/master_bench.py --causal --hkv 4      # [cuda] causal GQA fwd
    scripts/master_bench.py --varlen              # [cuda] packed varlen
    scripts/master_bench.py --full                # multi-shape sweep (all)
    scripts/master_bench.py --seq 4096 --head-dim 128   # [metal/rocm] one shape
    scripts/master_bench.py --profile mojo        # [metal] xctrace recording
    scripts/master_bench.py --gate                # [metal] mojo ratchet gate
    scripts/master_bench.py --no-prof --no-asm    # timing-only fast loop
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
BENCH_FA4 = str(SCRIPTS / "bench_fa4.py")
BENCH_ROCM = str(SCRIPTS / "bench_rocm.py")
# Absolute path: profiler phases re-invoke the bench under `sudo -E`, whose
# reset PATH would not otherwise find a bare `uv`.
UV = shutil.which("uv") or "uv"

_BOLD, _RST, _YEL, _RED, _GRN = (
    "\033[1m", "\033[0m", "\033[33m", "\033[31m", "\033[32m",
)
VERBOSE = False


# ==========================================================================
# Shared scaffolding (identical convention across all backends).
# ==========================================================================


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


def run(cmd, *, env=None, capture=False, cwd=None):
    if VERBOSE:
        print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _spread_pct(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    lo, hi = min(vals), max(vals)
    return (hi - lo) / lo * 100.0 if lo else 0.0


def emit_summary(summary: dict) -> None:
    print("===AGENT-SUMMARY===")
    print(json.dumps(summary))
    print("===END-AGENT-SUMMARY===")


# ==========================================================================
# Backend detection.
# ==========================================================================

_TORCH_PROBE = (
    "import torch;"
    "print('rocm' if (torch.cuda.is_available() and getattr(torch.version,'hip',None))"
    " else ('cuda' if torch.cuda.is_available() else 'cpu'))"
)


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


def detect_backend(requested: str) -> str:
    """Resolve the backend. `requested` is one of auto/cuda/rocm/metal/cpu."""
    if requested != "auto":
        return requested
    if sys.platform == "darwin":
        return "metal"  # Metal is the only GPU path on macOS
    # NVIDIA fast path (no torch import needed).
    if shutil.which("nvidia-smi") and nvidia_smi("name"):
        return "cuda"
    # Probe torch in the venv to distinguish rocm (HIP) from a CPU-only box.
    probe = run([UV, "run", "python", "-c", _TORCH_PROBE], capture=True)
    got = (probe.stdout or "").strip().splitlines()
    if probe.returncode == 0 and got:
        return got[-1].strip()
    warn("torch probe failed — falling back to CPU mode")
    return "cpu"


# ==========================================================================
# CUDA backend (mojo vs FlashAttention-4).
# ==========================================================================

CU_CANON = "2,8192,16,128"
CU_FULL_SHAPES = [
    "2,8192,16,128",  # canonical (many-wave, compute-bound)
    "1,4096,16,128",  # half batch, half seq
    "4,4096,16,128",  # more batch
    "1,2048,32,128",  # short seq, more heads
    "2,8192,32,64",   # hdim64
]

# bf16/fp16 dense tensor-core peak (TFLOP/s) by device-name substring. Longest
# match wins; FLASH_ATTN_MOJO_PEAK_TFLOPS overrides.
_CU_PEAK_TFLOPS = {
    "H100 NVL": 835.0, "H100 PCIe": 756.0, "H200": 989.0, "H100": 989.0,
    "A100": 312.0, "L40S": 362.0, "RTX 5090": 838.0, "RTX 4090": 330.0,
}

_CU_NCU_METRICS = [
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__waves_per_multiprocessor",
]
_CU_NCU_LAUNCH_SKIP = 3
_CU_NCU_LAUNCH_COUNT = 5
_CU_RESULT_RE = re.compile(r"us=([0-9.]+)\s+tflops=([0-9.]+)")


def cu_peak_tflops(gpu: str) -> float | None:
    override = os.environ.get("FLASH_ATTN_MOJO_PEAK_TFLOPS")
    if override:
        return float(override)
    for name in sorted(_CU_PEAK_TFLOPS, key=len, reverse=True):
        if name in gpu:
            return _CU_PEAK_TFLOPS[name]
    return None


def _cu_fatal_lock(reason: str) -> NoReturn:
    print(f"{_RED}[FAIL]{_RST} GPU clock lock unavailable — refusing to run unlocked.")
    print(f"{_RED}[FAIL]{_RST} ({reason})")
    print(f"{_RED}[FAIL]{_RST} pass --no-lock to run unlocked anyway (dev mode).")
    raise SystemExit(1)


def cu_lock_clocks(enabled: bool) -> tuple[str, bool]:
    section("(a) lock clocks")
    if not enabled:
        warn("--no-lock: clocks NOT locked (dev mode — numbers are not comparable).")
        return "unlocked", False

    def sudo(a) -> bool:
        return subprocess.run(
            ["sudo", "-n", "nvidia-smi", *a], capture_output=True, text=True
        ).returncode == 0

    max_sm = nvidia_smi("clocks.max.sm")
    if max_sm and sudo(["-pm", "1"]) and sudo([f"--lock-gpu-clocks={max_sm},{max_sm}"]):
        max_mem = nvidia_smi("clocks.max.mem")
        if max_mem:
            sudo([f"--lock-memory-clocks={max_mem},{max_mem}"])  # best-effort
        print(f"locked SM clock to {max_sm} MHz (resets on exit)")
        return max_sm, True
    _cu_fatal_lock("passwordless `sudo -n nvidia-smi -pm 1 --lock-gpu-clocks=…` unavailable")


def cu_unlock_clocks() -> None:
    for a in (["--reset-gpu-clocks"], ["--reset-memory-clocks"]):
        subprocess.run(["sudo", "-n", "nvidia-smi", *a], capture_output=True)


def cu_variant(args) -> dict:
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
        "cflag": cflag, "fa4_ptx": fa4_ptx, "mojo_ptx": mojo_ptx,
        "fa4_filter": fa4_filter, "mojo_filter": mojo_filter,
    }


def cu_bench_cmd(args, var, impl, shape, *, extra=()):
    return [
        UV, "run", "python", BENCH_FA4,
        "--impl", impl, "--kind", args.kind, "--shape", shape,
        "--iters", str(args.iters), *var["cflag"], *extra,
    ]


def cu_parse_results(text: str) -> list[tuple[float, float]]:
    out = []
    for line in text.splitlines():
        if line.startswith("RESULT"):
            m = _CU_RESULT_RE.search(line)
            if m:
                out.append((float(m.group(1)), float(m.group(2))))
    return out


def cu_correctness(args, var, clean: bool) -> None:
    section("(b) clear JIT cache + recompile + correctness")
    if clean:
        cache = Path("~/.cache/flash_attn_mojo").expanduser()
        if VERBOSE:
            print(f"clearing JIT cache {cache} (mojo compiler cache untouched)")
        shutil.rmtree(cache, ignore_errors=True)
    else:
        print("(--no-clean: keeping JIT cache)")
    if run(cu_bench_cmd(args, var, "mojo", args.shape, extra=["--check-only"])).returncode != 0:
        Gate.fail("correctness failed (see pytest output above)")
    else:
        print(f"{_GRN}correctness OK{_RST}")


def _cu_roofline(tflops: float, peak: float | None) -> tuple[str, str, float | None]:
    if not peak:
        return ("unknown", "set FLASH_ATTN_MOJO_PEAK_TFLOPS for a %-of-peak", None)
    pct = 100.0 * tflops / peak
    if pct >= 80:
        return ("compute-bound (near-peak)",
                "at the tensor-core roofline — the FA4 parity race is the only lever", pct)
    if pct >= 50:
        return ("compute-bound",
                "healthy TC utilization — chase the FA4 gap via the PTX/SASS op-mix", pct)
    if pct >= 25:
        return ("partial",
                "mid utilization — check occupancy / waves / pipeline stalls in (d)", pct)
    return ("latency/occupancy-bound",
            "small wave — grow B*S or inspect scheduler stalls (ncu source page)", pct)


def cu_bench(args, var, shapes, peak, gate: bool) -> list[dict]:
    section(f"(c) kernel-time bench vs FA4  (min of {args.runs} runs/impl, locked clock)")
    rows = []
    for shape in shapes:
        row = {"shape": shape}
        for impl in ("fa4", "mojo"):
            r = run(cu_bench_cmd(args, var, impl, shape, extra=["--runs", str(args.runs)]),
                    capture=True)
            if r.stderr and VERBOSE:
                sys.stderr.write(r.stderr)
            res = cu_parse_results(r.stdout or "")
            if r.returncode != 0 or not res:
                Gate.fail(f"{impl} bench failed on {shape}")
                if r.stderr:
                    sys.stderr.write(r.stderr[-800:])
                row[impl] = None
                continue
            us = [u for u, _ in res]
            tf = [t for _, t in res]
            row[impl] = {"us": min(us), "spread": _spread_pct(us), "tflops": max(tf)}
        rows.append(row)

    hdr = (f"  {'shape':>16} | {'mojo us':>9} | {'fa4 us':>9} | {'ratio':>7} | "
           f"{'spread':>7} | {'mojo TF/s':>9} | verdict")
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
                Gate.fail(f"perf regression on {row['shape']}: {ratio:.3f}x "
                          f"(gap {gap:.1f}% > spread {spread:.1f}%)")
        print(f"  {row['shape']:>16} | {mojo['us']:9.1f} | {fa4['us']:9.1f} | "
              f"{ratio:6.3f}x | {spread:6.1f}% | {mojo['tflops']:9.1f} | {verdict}")
    print("  " + "-" * (len(hdr) - 2))
    if worst:
        tail = "(<=1.03x everywhere)" if worst <= 1.03 else "(a shape exceeds 1.03x)"
        print(f"  worst mojo/fa4 ratio: {worst:.3f}x  {tail}")

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
        regime, hint, pct = _cu_roofline(mojo["tflops"], peak)
        row["roofline"] = {"regime": regime, "hint": hint, "pct_peak": pct}
        pct_s = f"{pct:.0f}%" if pct is not None else "?"
        print(f"  {row['shape']:>16} | {mojo['tflops']:9.1f} | {pct_s:>6} | {regime}")
        if regime not in ("compute-bound (near-peak)",):
            hints.append((row["shape"], hint))
    for shape, hint in hints:
        print(f"    {shape}: {hint}")
    return rows


def _cu_ncu_cmd() -> list[str] | None:
    ncu = shutil.which("ncu")
    if ncu:
        return [ncu]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "exec", "--spec", "nsight-compute=2024.3.2", "--", "ncu"]
    return None


def _cu_needs_sudo() -> bool:
    try:
        params = Path("/proc/driver/nvidia/params").read_text()
    except OSError:
        return False
    return any(ln.strip() == "RmProfilingAdminOnly: 1" for ln in params.splitlines())


def _cu_parse_ncu_csv(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if '"Metric Name"' in ln and '"Metric Value"' in ln), None)
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


def _cu_ncu_capture(ncu, sudo, args, var, impl, filt) -> dict[str, dict]:
    cmd = [
        *sudo, *ncu, "--target-processes", "all",
        "--kernel-name", f"regex:{filt}",
        "--launch-skip", str(_CU_NCU_LAUNCH_SKIP),
        "--launch-count", str(_CU_NCU_LAUNCH_COUNT),
        "--metrics", ",".join(_CU_NCU_METRICS), "--csv",
        UV, "run", "python", BENCH_FA4,
        "--impl", impl, "--kind", args.kind, "--shape", args.shape,
        "--profile", "--iters", str(_CU_NCU_LAUNCH_SKIP + _CU_NCU_LAUNCH_COUNT + 2),
        "--warmup", "3", *var["cflag"],
    ]
    if VERBOSE:
        print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE)
    if proc.returncode != 0:
        warn(f"ncu capture failed for {impl} (perf-counter perms?)")
        return {}
    return _cu_parse_ncu_csv(proc.stdout or "")


def cu_profiler(args, var, enabled: bool) -> None:
    if not enabled:
        skip("(d) ncu deep profiler", "--no-prof")
        return
    section("(d) ncu deep profiler — fa4 vs mojo")
    ncu = _cu_ncu_cmd()
    if ncu is None:
        warn("ncu not found and pixi unavailable — skipping deep profiling.")
        return
    sudo = []
    if _cu_needs_sudo():
        if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
            sudo = ["sudo", "-E"]
        else:
            warn("RmProfilingAdminOnly=1 and no passwordless sudo — skipping ncu.")
            return
    fa4 = _cu_ncu_capture(ncu, sudo, args, var, "fa4", var["fa4_filter"])
    mojo = _cu_ncu_capture(ncu, sudo, args, var, "mojo", var["mojo_filter"])
    if not fa4 and not mojo:
        return

    def cell(d: dict, name: str) -> str:
        m = d.get(name)
        if not m or not m["vals"]:
            return "-"
        vals = m["vals"]
        return f"{sum(vals) / len(vals):.2f} (±{_spread_pct(vals):.0f}%)"

    hdr = f"    {'metric':<52} | {'unit':>7} | {'fa4':>16} | {'mojo':>16}"
    print(f"\n{hdr}")
    print("    " + "-" * (len(hdr) - 4))
    for name in _CU_NCU_METRICS:
        unit = (fa4.get(name) or mojo.get(name) or {}).get("unit", "")
        print(f"    {name:<52} | {unit:>7} | {cell(fa4, name):>16} | {cell(mojo, name):>16}")
    n = max([len(m["vals"]) for m in list(fa4.values()) + list(mojo.values())], default=0)
    print(f"    (mean ± run-to-run spread over {n} profiled launches per kernel)")


def _cu_find_ptxas() -> str | None:
    direct = shutil.which("ptxas")
    if direct:
        return direct
    for pat in (".venv/lib/python*/site-packages/torch/bin/ptxas",
                ".venv/lib/python*/site-packages/triton/backends/nvidia/bin/ptxas"):
        hits = sorted(REPO.glob(pat))
        if hits:
            return str(hits[0])
    return None


def cu_assembly(args, var, enabled: bool, refresh_fa4: bool) -> None:
    if not enabled:
        skip("(e/f/g) PTX dump + histogram + spill canary", "--no-asm")
        return
    mojo_ptx = var["mojo_ptx"]
    mojo_ptx.parent.mkdir(parents=True, exist_ok=True)

    section("(e) dump mojo PTX -> ptx/")
    env = {**os.environ, "MOJO_DUMP_PTX": str(mojo_ptx)}
    cmd = cu_bench_cmd(args, var, "mojo", args.shape, extra=["--iters", "1", "--warmup", "0"])
    if run(cmd, env=env, capture=not VERBOSE).returncode != 0 or not mojo_ptx.exists():
        Gate.fail("PTX dump failed")
        return
    print(f"mojo PTX: {mojo_ptx.relative_to(REPO)}")

    section("(g) ptxas -v spill / regalloc canary")
    ptxas = _cu_find_ptxas()
    if not ptxas:
        warn("ptxas not found (PATH or venv) — skipping spill canary")
    else:
        r = run([ptxas, "-arch=sm_90a", "-v", str(mojo_ptx)], capture=True)
        blob = (r.stderr or "") + (r.stdout or "")
        spill = sum(int(m.group(1)) for m in re.finditer(r"(\d+)\s+bytes spill stores", blob))
        regs = re.search(r"Used (\d+) registers", blob)
        smem = re.search(r"(\d+) bytes smem", blob)
        print(f"  registers: {regs.group(1) if regs else '?'} | "
              f"smem: {smem.group(1) if smem else '?'} B | spill stores: {spill} B")
        if spill > 0:
            Gate.fail(f"spill canary: {spill} bytes spilled (must be 0)")
        else:
            print(f"  {_GRN}no spills{_RST}")

    section("(f) PTX instruction-mix histogram: fa4 (reference) vs mojo")
    fa4_ptx = var["fa4_ptx"]
    if refresh_fa4:
        warn("--refresh-fa4-ptx: regenerate reference PTX via "
             "master_bench.sh --refresh-fa4-ptx (not automated here)")
    stats = str(SCRIPTS / "ptx_stats.py")
    if not fa4_ptx.exists():
        warn(f"no FA4 reference PTX at {fa4_ptx.relative_to(REPO)} — showing mojo op-mix only")
        run([UV, "run", "python", stats, str(mojo_ptx)])
    else:
        run([UV, "run", "python", stats, str(fa4_ptx), str(mojo_ptx)])


def cu_walltime(args, var) -> None:
    section("(h) end-to-end wall-clock (CUDA-event timed, launch+sync included)")
    rows = []
    for impl in ("fa4", "mojo"):
        r = run(cu_bench_cmd(args, var, impl, args.shape,
                             extra=["--walltime", "--runs", str(args.runs)]), capture=True)
        res = cu_parse_results(r.stdout or "")
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
        print("  (compare vs the (c) kernel-time ratio: a wall-clock-only "
              "regression is launch/dispatch overhead, not the kernel)")


def run_cuda(args) -> int:
    if not (shutil.which("nvidia-smi") and nvidia_smi("name")):
        print(f"{_RED}[FAIL]{_RST} --device cuda but no NVIDIA GPU detected.")
        return 1
    gpu = nvidia_smi("name")
    peak = cu_peak_tflops(gpu)
    var = cu_variant(args)
    shapes = CU_FULL_SHAPES if args.full else [args.shape]
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
    print(f"master_bench: backend=cuda device='{gpu}'{peak_note}")
    print(f"variant: {' + '.join(axes)}   shapes: {', '.join(shapes)}")

    clock, locked = cu_lock_clocks(not args.no_lock)
    if locked:
        def _on_signal(signum, _frame):
            cu_unlock_clocks()
            os._exit(128 + signum)
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGHUP, _on_signal)

    rows: list[dict] = []
    try:
        if args.no_check:
            skip("(b) correctness", "--no-check")
        else:
            cu_correctness(args, var, not args.no_clean)
        rows = cu_bench(args, var, shapes, peak, gate=not args.no_gate)
        cu_profiler(args, var, not args.no_prof)
        cu_assembly(args, var, not args.no_asm, args.refresh_fa4_ptx)
        if not args.no_walltime:
            cu_walltime(args, var)
        else:
            skip("(h) wall-clock", "--no-walltime")
    finally:
        if locked:
            cu_unlock_clocks()

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
    emit_summary({
        "backend": "cuda", "device": gpu, "peak_tflops": peak, "clock": clock,
        "variant": " + ".join(axes), "gate": "fail" if Gate.failed else "pass",
        "shapes": agent_rows,
    })
    if Gate.failed:
        print(f"{_RED}ISSUES — one or more gates failed above.{_RST}")
        return 1
    print(f"{_GRN}PASS — all gates green (kind={args.kind} variant={' + '.join(axes)}).{_RST}")
    return 0


# ==========================================================================
# Metal backend (mojo vs best-of {MFA, ccv}).
# ==========================================================================

MT_QUICK_SHAPES = [(1024, 128), (4096, 128)]  # (seq, head_dim)
MT_FULL_SHAPES = [
    (1024, 64), (1024, 128), (4096, 64), (4096, 128), (8192, 64), (8192, 128),
]
MT_CORRECTNESS_SHAPES = [(1024, 64), (1024, 128)]
MT_XCTRACE_TEMPLATE = "Metal System Trace"
_MT_BASELINE = SCRIPTS / "baselines" / "metal_fwd_kernel_us.json"


def mt_peak_tflops() -> float | None:
    env = os.environ.get("FLASH_ATTN_MOJO_METAL_PEAK_TFLOPS")
    if not env:
        return None
    try:
        return float(env)
    except ValueError:
        warn(f"ignoring non-numeric FLASH_ATTN_MOJO_METAL_PEAK_TFLOPS={env!r}")
        return None


def _mt_fwd_flops(seq, heads, head_dim, batch=1) -> int:
    return 4 * batch * seq * seq * head_dim * heads


def _mt_tflops(flops, us) -> float:
    return flops / (us * 1e6)


def mt_lock_clock(enabled: bool):
    section("(a) clock lock (induced GPU performance state)")
    if not enabled:
        print("skipped — --no-lock (bench self-times interleaved; drift cancels)")
        return "ambient", None
    from _apple_gpu_clock_lock import locked_template_path
    path = locked_template_path("Maximum")
    if path is not None:
        print(f"induced-max template ready: {path}")
        print("  (applies to the (d) xctrace recording; the (c) bench runs at "
              "ambient clock, interleaved)")
        return "induced-maximum", str(path)
    warn("could not build an induced-max template (Xcode internals may have "
         "changed — see scripts/_apple_gpu_clock_lock.py); profiler falls back "
         "to the ambient-clock named template")
    return "ambient", None


def _mt_lane_binary(impl: str) -> Path:
    return {
        "mojo": REPO / "bench/build/bench_mojo_metal",
        "mfa": REPO / "reference_air/mfa/bench_mfa/.build/release/bench_mfa",
        "ccv": REPO / "reference_air/ccv/bench_ccv/bench_ccv_attn",
    }[impl]


def mt_build_lanes(bm, impls, *, do_build: bool) -> None:
    section("build lanes")
    if not do_build:
        missing = [i for i in impls if not _mt_lane_binary(i).exists()]
        if not missing:
            print("skipped — --no-build (all lane binaries present)")
            return
        warn(f"--no-build but missing binaries: {missing} — building those")
        impls = missing
    try:
        bm.build_impls(impls, log=print)
    except subprocess.CalledProcessError as e:
        Gate.fail(f"lane build failed: {e}")
        raise SystemExit(1)
    print("built:", ", ".join(impls))


def mt_correctness(bm, impls, heads, dtype) -> bool:
    section("(b) correctness gate (fp32-reference --check)")
    ok = True
    for seq, head_dim in MT_CORRECTNESS_SHAPES:
        try:
            res = bm.bench_shape(impls, seq=seq, heads=heads, head_dim=head_dim,
                                 dtype=dtype, rounds=1, iters=2, warmup=1, check=True)
        except bm.BenchError as e:
            Gate.fail(f"correctness run failed (S={seq} D={head_dim}): {e}")
            ok = False
            continue
        tol = bm.CHECK_TOL[dtype]
        for i in impls:
            err = res[i]["check_err"]
            good = err is not None and err <= tol
            status = "no check" if err is None else f"max|err| {err:.2e}"
            mark = f"{_GRN}OK{_RST}" if good else f"{_RED}FAIL{_RST}"
            print(f"  S={seq:>5} D={head_dim:>3} {i:<5}: {status} vs tol {tol:.0e}  {mark}")
            if not good:
                Gate.fail(f"correctness: {i} S={seq} D={head_dim} {status} > tol {tol:.0e}")
                ok = False
    return ok


def _mt_roofline(ratio, baseline_impl, gap_pct):
    if ratio <= 1.03:
        return "at/above the best reference (ceiling)", None
    if ratio <= 1.5:
        return "near ceiling", f"close the last {gap_pct:.0f}% vs {baseline_impl}"
    if ratio <= 3.0:
        return "behind ceiling", f"{ratio:.2f}x off {baseline_impl}"
    hint = f"{ratio:.1f}x off {baseline_impl}"
    if ratio > 5.0:
        hint += " — v0 has 0 matrix ops; needs simdgroup_matrix MMA (METAL_PLAN v1 fight)"
    return "far behind ceiling", hint


def _mt_baseline_key(r: dict) -> str:
    return json.dumps({
        "v": 1, "fn": r.get("fn"), "seq": r.get("seq"), "heads": r.get("heads"),
        "head_dim": r.get("head_dim"), "dtype": r.get("dtype"), "device": platform.node(),
    }, sort_keys=True)


class _MtRatchet:
    """Per-shape best-time gate persisted in a JSON baseline (gitignored)."""

    def __init__(self, path: Path, tolerance: float):
        self.path = path
        self.tolerance = tolerance
        self.data: dict = {}
        self.dirty = False
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                self.data = data
            else:
                warn(f"baseline {path.name} is not a JSON object — starting fresh")
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as e:
            warn(f"baseline {path.name} unreadable ({e.__class__.__name__}) — fresh")

    def check(self, r, us, *, refresh, gate) -> str:
        key = _mt_baseline_key(r)
        prev = self.data.get(key)
        if prev is not None and not isinstance(prev, (int, float)):
            prev = None
        if refresh or prev is None:
            self.data[key] = us
            self.dirty = True
            return "baseline reseeded" if refresh else "baseline established"
        if us > prev * self.tolerance:
            msg = (f"mojo regression S{r['seq']}/D{r['head_dim']}: {us:.1f} us vs "
                   f"best {prev:.1f} us (>{(self.tolerance - 1) * 100:.0f}%; "
                   f"--refresh-baseline to accept)")
            if gate:
                Gate.fail(msg)
            else:
                warn(msg + " [not gated; pass --gate]")
            return f"REGRESSION vs best {prev:.1f} us"
        if us < prev:
            self.data[key] = us
            self.dirty = True
            return f"new best (was {prev:.1f} us)"
        return f"within tol of best {prev:.1f} us"

    def save(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            tmp.replace(self.path)
            print(f"  (baseline updated: {self.path.relative_to(REPO)})")


def mt_bench(bm, impls, shapes, heads, dtype, rounds, iters, dispatches, clock, *,
             gate, ratchet_refresh, jsonl) -> list[dict]:
    section("(c) GPU kernel-time bench (interleaved 3-way) vs best(mfa,ccv)")
    refs = [i for i in impls if i != "mojo"]
    have_mojo = "mojo" in impls

    rows: list[dict] = []
    for seq, head_dim in shapes:
        try:
            res = bm.bench_shape(impls, seq=seq, heads=heads, head_dim=head_dim,
                                 dtype=dtype, rounds=rounds, iters=iters,
                                 dispatches=dispatches, jsonl=jsonl)
        except bm.BenchError as e:
            Gate.fail(f"bench failed (S={seq} D={head_dim}): {e}")
            continue
        flops = _mt_fwd_flops(seq, heads, head_dim)
        row: dict = {
            "fn": "fwd", "seq": seq, "heads": heads, "head_dim": head_dim,
            "dtype": dtype, "clock": clock,
            "impls": {i: {"min_us": res[i]["min"], "median_us": res[i]["median"],
                          "spread_pct": res[i]["spread_pct"],
                          "tflops": _mt_tflops(flops, res[i]["min"])} for i in impls},
        }
        ref_mins = {r: res[r]["min"] for r in refs}
        if ref_mins:
            baseline_impl = min(ref_mins, key=ref_mins.get)
            row["baseline_impl"] = baseline_impl
            row["baseline_us"] = ref_mins[baseline_impl]
            row["baseline_tflops"] = _mt_tflops(flops, ref_mins[baseline_impl])
        rows.append(row)

    if not rows:
        Gate.fail("no parseable bench results")
        return rows

    hdr = (f"  {'shape':>13} | {'mojo us':>9} | {'mfa us':>9} | {'ccv us':>9} | "
           f"{'base':>4} | {'ratio':>7} | {'TFLOP/s':>8} | {'%ceil':>6} | verdict")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    ratchet = _MtRatchet(_MT_BASELINE, 1.10 if clock == "induced-maximum" else 1.20)
    for row in rows:
        seq, head_dim = row["seq"], row["head_dim"]
        im = row["impls"]
        mojo_us = im.get("mojo", {}).get("min_us", math.nan)
        mfa_us = im.get("mfa", {}).get("min_us", math.nan)
        ccv_us = im.get("ccv", {}).get("min_us", math.nan)
        base_impl = row.get("baseline_impl")
        base_us = row.get("baseline_us")
        shape_lbl = f"S{seq}/D{head_dim}"

        if have_mojo and base_us:
            ratio = mojo_us / base_us
            row["ratio_over_baseline"] = ratio
            for r in refs:
                row[f"ratio_over_{r}"] = mojo_us / im[r]["min_us"]
            gap = abs(ratio - 1) * 100
            spread = max(im["mojo"]["spread_pct"], im[base_impl]["spread_pct"])
            if gap <= 3:
                verdict = "parity"
            elif ratio < 1 and gap >= spread:
                verdict = f"{_GRN}FASTER{_RST}"
            elif gap < spread:
                verdict = "noise"
            else:
                verdict = f"{_RED}slower{_RST}"
            regime, hint = _mt_roofline(ratio, base_impl, gap)
            row["regime"], row["hint"] = regime, hint
            tflops = im["mojo"]["tflops"]
            pct_ceil = tflops / row["baseline_tflops"] * 100
            row["pct_of_ceiling"] = pct_ceil
            ratio_s = f"{ratio:6.2f}x"
            pct_s = f"{pct_ceil:5.0f}%"
        else:
            ratio_s, pct_s, verdict = "  n/a", "  n/a", "no-mojo"
            tflops = im.get(base_impl, {}).get("tflops", math.nan) if base_impl else math.nan
        print(f"  {shape_lbl:>13} | {mojo_us:9.1f} | {mfa_us:9.1f} | {ccv_us:9.1f} | "
              f"{(base_impl or '?'):>4} | {ratio_s:>7} | {tflops:8.2f} | {pct_s:>6} | {verdict}")

        if have_mojo and not math.isnan(mojo_us):
            row["ratchet"] = ratchet.check(row, mojo_us, refresh=ratchet_refresh, gate=gate)

    print("  " + "-" * (len(hdr) - 2))
    if have_mojo:
        worst = max((r.get("ratio_over_baseline", 0.0) for r in rows), default=0.0)
        print(f"  worst mojo/best-ref ratio: {worst:.2f}x  "
              f"(ratio is reported, not gated — mojo races the MMA references; "
              f"regression gate is the mojo ratchet under --gate)")
    _mt_print_roofline_hints(rows)
    ratchet.save()
    return rows


def _mt_print_roofline_hints(rows) -> None:
    hints = [(f"S{r['seq']}/D{r['head_dim']}", r["hint"]) for r in rows if r.get("hint")]
    if not hints:
        return
    peak = mt_peak_tflops()
    peak_note = f" (hw peak = {peak:.0f} TFLOP/s)" if peak else ""
    print(f"\n  compute roofline{peak_note}: attention is matmul-bound; "
          f"the best reference is the empirical ceiling.")
    for shape, hint in hints:
        print(f"    {shape}: {hint}")
    if peak:
        for r in rows:
            if r.get("baseline_tflops"):
                pct = r["baseline_tflops"] / peak * 100
                print(f"    {('S' + str(r['seq']) + '/D' + str(r['head_dim'])):>10}: "
                      f"best ref {r['baseline_tflops']:.0f} TFLOP/s = {pct:.0f}% of hw peak")


# --- true GPU kernel time for the mojo lane (xctrace) --------------------
# The reference CLIs self-report command-buffer GPU time (gpuEndTime -
# gpuStartTime); the mojo lane can only wall-clock enqueue+sync, so it
# eats ~160 us/dispatch of overhead the refs don't. To compare
# kernel-to-kernel we recover the mojo lane's true per-dispatch GPU time
# from an xctrace 'Metal System Trace' (the ONE thing on this toolchain
# that sees per-encoder GPU intervals).

_MT_KERNEL_TARGET_US = 1_200_000  # size each capture to ~1.2 s of GPU work
                                  # so DVFS ramps to a stable top clock


def _mt_capture_kernel_us(seq, head_dim, heads, wall_us, template) -> float | None:
    """xctrace-record the mojo lane at one shape and return its true
    steady-state per-dispatch GPU kernel time (us), or None on failure.

    The capture is sized from the wall time so total GPU work ~1.2 s —
    short shapes need hundreds of dispatches to make the GPU commit to
    its top DVFS clock (else per-dispatch time is dominated by a low
    ramp clock and over-reads). xctrace SIGSEGVs on finalize ~1/3 of
    the time but writes the trace first, and its export intermittently
    throws 'missing template' — both are retry-recoverable.
    """
    import tempfile
    sys.path.insert(0, str(SCRIPTS))
    from xctrace_gpu_intervals import steady_state_kernel_us

    total = max(12, min(400, round(_MT_KERNEL_TARGET_US / max(wall_us, 1.0))))
    disp = 4
    iters = max(1, -(-total // disp))  # ceil
    warmup = max(2, min(20, iters // 5))
    binary = _mt_lane_binary("mojo")
    argv = [str(binary), "--seq", str(seq), "--head-dim", str(head_dim),
            "--heads", str(heads), "--iters", str(iters), "--warmup", str(warmup),
            "--dispatches", str(disp)]
    for _ in range(5):
        trace = Path(tempfile.mkdtemp()) / "k.trace"
        rec = ["xctrace", "record", "--template", template or MT_XCTRACE_TEMPLATE,
               "--output", str(trace), "--launch", "--", *argv]
        subprocess.run(rec, capture_output=True, text=True)
        if not trace.exists():
            continue
        try:
            us = steady_state_kernel_us(str(trace), binary.name)
        except RuntimeError:
            us = None
        shutil.rmtree(trace.parent, ignore_errors=True)
        if us:
            return us
    return None


def mt_kernel_time(rows, heads, template, *, enabled: bool) -> None:
    """Recover the mojo lane's true GPU kernel time per shape (default
    on for metal) and print a kernel-to-kernel comparison table.

    Augments each row in place: ``impls['mojo']['kernel_us']`` and
    ``kernel_ratio_over_baseline`` (mojo kernel us / best-ref GPU us).
    Falls back to the wall time for any shape whose capture fails,
    flagged in the table.
    """
    section("(c2) true GPU kernel time — mojo lane via xctrace (refs "
            "self-report GPU time)")
    if not enabled:
        skip("kernel-time", "--no-kernel-time (mojo column stays wall-clock)")
        return
    if shutil.which("xctrace") is None:
        warn("xctrace not found (needs Xcode) — mojo column stays wall-clock")
        return

    hdr = (f"  {'shape':>13} | {'mojo kern':>10} | {'mojo wall':>10} | "
           f"{'best ref':>10} | {'ref':>4} | {'kern ratio':>10} | verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    any_ok = False
    for row in rows:
        im = row["impls"]
        if "mojo" not in im:
            continue
        seq, head_dim = row["seq"], row["head_dim"]
        wall_us = im["mojo"]["min_us"]
        kern = _mt_capture_kernel_us(seq, head_dim, heads, wall_us, template)
        base_impl = row.get("baseline_impl")
        base_us = row.get("baseline_us")
        shape_lbl = f"S{seq}/D{head_dim}"
        if kern is None:
            print(f"  {shape_lbl:>13} | {'FAILED':>10} | {wall_us:9.1f}u | "
                  f"{(base_us or float('nan')):9.1f}u | {(base_impl or '?'):>4} | "
                  f"{'(wall)':>10} | xctrace unavailable — using wall")
            im["mojo"]["kernel_us"] = None
            continue
        any_ok = True
        im["mojo"]["kernel_us"] = kern
        if base_us:
            ratio = kern / base_us
            row["kernel_ratio_over_baseline"] = ratio
            gap = abs(ratio - 1) * 100
            if gap <= 3:
                verdict = "parity"
            elif ratio < 1:
                verdict = f"{_GRN}FASTER{_RST}"
            else:
                verdict = f"{_RED}slower{_RST}"
            ratio_s = f"{ratio:9.2f}x"
        else:
            ratio_s, verdict = "  n/a", "no ref"
        print(f"  {shape_lbl:>13} | {kern:9.1f}u | {wall_us:9.1f}u | "
              f"{(base_us or float('nan')):9.1f}u | {(base_impl or '?'):>4} | "
              f"{ratio_s:>10} | {verdict}")
    print("  " + "-" * (len(hdr) - 2))
    if any_ok:
        worst = max((r.get("kernel_ratio_over_baseline", 0.0) for r in rows),
                    default=0.0)
        print(f"  worst mojo-kernel/best-ref ratio: {worst:.2f}x  "
              f"(true GPU time, both sides kernel-only)")


def mt_profiler(impl, template, heads) -> None:
    if not impl:
        skip("(d) xctrace profile", "pass --profile <impl> to record one")
        return
    section(f"(d) xctrace Metal System Trace: {impl} (S=4096 D=128)")
    seq, head_dim = 4096, 128
    argv = {
        "mojo": [str(_mt_lane_binary("mojo")), "--seq", str(seq), "--head-dim",
                 str(head_dim), "--heads", str(heads), "--iters", "3", "--warmup", "1"],
        "mfa": [str(_mt_lane_binary("mfa")), "--seq", str(seq), "--head-dim",
                str(head_dim), "--heads", str(heads), "--iters", "3", "--warmup", "1"],
        "ccv": [str(_mt_lane_binary("ccv")), "--b", "1", "--r", str(seq), "--c", str(seq),
                "--hq", str(heads), "--hk", str(heads), "--d", str(head_dim),
                "--iterations", "3", "--warmup", "1"],
    }.get(impl)
    if argv is None:
        warn(f"unknown --profile impl {impl!r}")
        return
    import tempfile
    # xctrace SIGSEGVs intermittently while finalizing (~1 in 3) but writes the
    # .trace before crashing — retry, accept the first parseable trace.
    intervals = SCRIPTS / "xctrace_gpu_intervals.py"
    last = ""
    for _ in range(3):
        trace = Path(tempfile.mkdtemp()) / "bench.trace"
        rec = ["xctrace", "record", "--template", template or MT_XCTRACE_TEMPLATE,
               "--output", str(trace), "--launch", "--", *argv]
        r = subprocess.run(rec, capture_output=True, text=True)
        last = f"rc={r.returncode}: {(r.stdout or '')[-300:]}{(r.stderr or '')[-300:]}"
        if not trace.exists():
            continue
        parsed = subprocess.run(
            [sys.executable, str(intervals), str(trace), "--process", Path(argv[0]).name],
            capture_output=True, text=True)
        if parsed.returncode == 0 and parsed.stdout.strip():
            print(parsed.stdout, end="")
            print(f"  trace kept at: {trace} (open in Instruments for HW counters — GUI-only)")
            return
    warn(f"xctrace profile unavailable after 3 attempts (last {last})")


def mt_introspect(shapes, refs, *, enabled: bool) -> None:
    if not enabled:
        skip("(e) AIR introspection", "--no-asm")
        return
    section("(e) AIR op-mix: mojo vs references (static IR, whole kernel)")
    air_dir = REPO / "air"
    air_dir.mkdir(exist_ok=True)
    for D in sorted({d for _, d in shapes}):
        src = Path(f"/tmp/mojo_fwd_metal_d{D}.air.ll")
        dst = air_dir / f"mojo_fwd_metal_d{D}.air.ll"
        if src.exists():
            dst.write_bytes(src.read_bytes())
        if not dst.exists():
            warn(f"no mojo AIR for d{D} (run the bench first); skipping op-mix")
            continue
        for ref in refs:
            ref_air = REPO / f"reference_air/{ref}/fwd_d{D}.air"
            if not ref_air.exists():
                continue
            print(f"\n--- d{D}: {ref} -> mojo (top 12 by |delta|) ---")
            subprocess.run([sys.executable, str(SCRIPTS / "air_opmix.py"),
                            str(ref_air), str(dst), "--top", "12"])


def _mt_num(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def _mt_agent_row(r: dict) -> dict:
    im = r.get("impls", {})
    return {
        "fn": r["fn"], "seq": r["seq"], "heads": r["heads"], "head_dim": r["head_dim"],
        "dtype": r["dtype"],
        "mojo_us": _mt_num(im.get("mojo", {}).get("min_us")),
        "mojo_kernel_us": _mt_num(im.get("mojo", {}).get("kernel_us")),
        "mfa_us": _mt_num(im.get("mfa", {}).get("min_us")),
        "ccv_us": _mt_num(im.get("ccv", {}).get("min_us")),
        "baseline_impl": r.get("baseline_impl"),
        "baseline_us": _mt_num(r.get("baseline_us")),
        "ratio_over_baseline": _mt_num(r.get("ratio_over_baseline")),
        "kernel_ratio_over_baseline": _mt_num(r.get("kernel_ratio_over_baseline")),
        "ratio_over_mfa": _mt_num(r.get("ratio_over_mfa")),
        "ratio_over_ccv": _mt_num(r.get("ratio_over_ccv")),
        "mojo_tflops": _mt_num(im.get("mojo", {}).get("tflops")),
        "baseline_tflops": _mt_num(r.get("baseline_tflops")),
        "pct_of_ceiling": _mt_num(r.get("pct_of_ceiling")),
        "spread_pct": _mt_num(im.get("mojo", {}).get("spread_pct")),
        "regime": r.get("regime"), "hint": r.get("hint"),
    }


def run_metal(args) -> int:
    if sys.platform != "darwin":
        print(f"{_RED}[FAIL]{_RST} --device metal targets the Apple GPU (darwin only).")
        return 1
    sys.path.insert(0, str(SCRIPTS))
    import bench_metal as bm  # noqa: E402

    impls = [i.strip() for i in (args.impls or "mojo,mfa,ccv").split(",") if i.strip()]
    for i in impls:
        if i not in bm.IMPLS:
            print(f"{_RED}[FAIL]{_RST} unknown impl '{i}' (have: {', '.join(bm.IMPLS)})")
            return 1
    refs = [i for i in impls if i != "mojo"]
    dtype = args.dtype if args.dtype in bm.CHECK_TOL else "fp16"

    if args.seq or args.head_dim:
        if not (args.seq and args.head_dim):
            print(f"{_RED}[FAIL]{_RST} --seq and --head-dim must be given together.")
            return 1
        shapes = [(args.seq, args.head_dim)]
    else:
        shapes = MT_FULL_SHAPES if args.full else MT_QUICK_SHAPES
    tier = "full" if args.full else "quick"
    rounds = args.rounds if args.rounds is not None else (3 if args.full else 2)
    iters = args.iters if args.iters is not None else 5

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip() or "Apple GPU"
    macos = (platform.mac_ver()[0] or "?").split(".")[0]
    print(f"master_bench: backend=metal fn=fwd tier={tier} dtype={dtype} "
          f"device='{chip}' macos{macos} impls={','.join(impls)}")

    clock, template = mt_lock_clock(not args.no_lock)
    signal.signal(signal.SIGTERM, lambda s, f: os._exit(128 + s))

    mt_build_lanes(bm, impls, do_build=not args.no_build)

    if args.no_check:
        skip("(b) correctness", "--no-check")
    elif not mt_correctness(bm, impls, args.heads or 16, dtype):
        print(f"\n{_RED}ISSUES — correctness gate failed{_RST}")
        return 1

    rows = mt_bench(bm, impls, shapes, args.heads or 16, dtype, rounds, iters,
                    args.dispatches, clock, gate=args.gate,
                    ratchet_refresh=args.refresh_baseline, jsonl=args.jsonl)

    if "mojo" in impls:
        mt_kernel_time(rows, args.heads or 16, template,
                       enabled=not args.no_kernel_time)

    mt_profiler(args.profile, template, args.heads or 16)
    mt_introspect(shapes, refs, enabled=not args.no_asm)

    section("summary")
    agent_rows = [_mt_agent_row(r) for r in rows]
    if agent_rows:
        emit_summary({
            "backend": "metal", "device": chip, "tier": tier, "dtype": dtype,
            "baseline": "best-of(mfa,ccv) per shape",
            "gate": "fail" if Gate.failed else "pass", "shapes": agent_rows,
        })
    if Gate.failed:
        print(f"\n{_RED}ISSUES — one or more gates failed above (tier={tier}){_RST}")
        return 1
    print(f"\n{_GRN}PASS — all gates green (backend=metal fn=fwd tier={tier}){_RST}")
    return 0


# ==========================================================================
# ROCm backend (mojo v0 vs CK-flash).
# ==========================================================================

_RC_MOJO_BIN = REPO / "bench/build/bench_mojo_rocm"
_RC_RESULT_RE = re.compile(r"us=([0-9.]+)")
_RC_TFLOPS_RE = re.compile(r"tflops=([0-9.]+)")
# A v0 hand-vectorized kernel checked against a strided fp32 reference: gate
# only egregious breakage (a broken softmax/mask shows at >=1e-2).
_RC_CHECK_TOL = 5e-2


def _rc_have_flash_attn() -> bool:
    return run([UV, "run", "python", "-c", "import flash_attn"], capture=True).returncode == 0


def _rc_kernel_only_us(csv_path: Path) -> float | None:
    """Mean mojo kernel-only time from a rocprofv3 --kernel-trace CSV (drop the
    first dispatch as warmup) — removes mojo's dispatch-overhead asymmetry."""
    try:
        rows = list(csv.DictReader(csv_path.open()))
    except OSError:
        return None
    durs = []
    for r in rows:
        try:
            durs.append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
        except (KeyError, ValueError):
            continue
    if len(durs) > 1:
        durs = durs[1:]
    return sum(durs) / len(durs) if durs else None


def run_rocm(args) -> int:
    seq = 1024 if args.quick else (args.seq or 4096)
    hdim = args.head_dim or 128
    heads = 8 if args.quick else (args.heads or 16)
    batch = args.batch or 1
    iters = 5 if args.quick else (args.iters or 20)
    dtype = args.dtype if args.dtype in ("fp16", "bf16") else "fp16"
    print(f"master_bench: backend=rocm fn=fwd device='{platform.node()}' "
          f"shape=B{batch}/S{seq}/H{heads}/D{hdim} dtype={dtype}")

    if not _rc_have_flash_attn():
        Gate.fail("flash_attn (Composable Kernel backend) is not installed — it is the "
                  "required AMD reference baseline. Build it per scripts/README.md.")
        emit_summary({"backend": "rocm", "gate": "fail",
                      "error": "flash_attn (CK) not installed"})
        print(f"{_RED}ISSUES — CK reference missing.{_RST}")
        return 1

    # (1) build the mojo v0 kernel from source.
    section("(1) build mojo v0 kernel (recompile from source)")
    (REPO / "bench/build").mkdir(parents=True, exist_ok=True)
    (REPO / "asm").mkdir(parents=True, exist_ok=True)
    if args.no_build and _RC_MOJO_BIN.exists():
        print("skipped — --no-build (binary present)")
    else:
        mojo = REPO / ".venv/bin/mojo"
        b = run([str(mojo), "build", "bench/bench_mojo_rocm.mojo", "-o", str(_RC_MOJO_BIN)],
                cwd=str(REPO))
        if b.returncode != 0:
            Gate.fail("mojo v0 build failed")
            emit_summary({"backend": "rocm", "gate": "fail", "error": "mojo build failed"})
            return 1

    # (2) mojo v0: run + fp32 correctness.
    section(f"(2) mojo v0 (S{seq} x H{heads} x D{hdim}): run + fp32 correctness")
    mr = run([str(_RC_MOJO_BIN), "--seq", str(seq), "--head-dim", str(hdim),
              "--heads", str(heads), "--iters", str(iters), "--check"], capture=True)
    if mr.stdout:
        sys.stdout.write(mr.stdout)
    mojo = None
    for line in (mr.stdout or "").splitlines():
        if line.startswith("{"):
            try:
                mojo = json.loads(line)
            except json.JSONDecodeError:
                pass
    if mr.returncode != 0 or not mojo:
        Gate.fail("mojo v0 run failed (no JSON line)")
        if mr.stderr:
            sys.stderr.write(mr.stderr[-800:])
    else:
        err = mojo.get("check_max_error")
        if err is not None and err > _RC_CHECK_TOL:
            Gate.fail(f"mojo v0 correctness: max|err| {err:.2e} > tol {_RC_CHECK_TOL:.0e}")
        elif err is not None:
            print(f"{_GRN}mojo correctness OK{_RST} (max|err| {err:.2e} vs tol {_RC_CHECK_TOL:.0e})")

    # (2b) CK reference: bench (kernel-only via roctracer) + correctness.
    section("(2b) CK reference (kernel-only): bench + correctness")
    cr = run([UV, "run", "python", BENCH_ROCM, "--batch", str(batch), "--seq", str(seq),
              "--heads", str(heads), "--head-dim", str(hdim), "--iters", str(iters),
              "--dtype", dtype, "--check"], capture=True)
    if cr.stdout:
        sys.stdout.write(cr.stdout)
    ref_us = ref_tflops = None
    for line in (cr.stdout or "").splitlines():
        if line.startswith("RESULT"):
            mu, mt = _RC_RESULT_RE.search(line), _RC_TFLOPS_RE.search(line)
            if mu:
                ref_us = float(mu.group(1))
            if mt:
                ref_tflops = float(mt.group(1))
    if cr.returncode != 0 or ref_us is None:
        Gate.fail("CK reference bench failed")
        if cr.stderr:
            sys.stderr.write(cr.stderr[-800:])

    # (3) AMDGCN ISA dump -> asm/.
    section("(3) AMDGCN ISA dump -> asm/")
    isa_src = Path(f"/tmp/mojo_fwd_rocm_d{hdim}.s")
    isa_dst = REPO / "asm" / f"mojo_fwd_rocm_d{hdim}.s"
    if isa_src.exists():
        isa_dst.write_bytes(isa_src.read_bytes())
        print(f"copied {isa_src} -> {isa_dst.relative_to(REPO)}")
    else:
        warn(f"{isa_src} not found (run the mojo bench first)")

    # (4) GCN instruction mix + resources (spill canary + MFMA count).
    if args.no_asm:
        skip("(4) GCN op-mix", "--no-asm")
    else:
        section("(4) GCN instruction mix + resources (mojo v0 kernel)")
        if isa_dst.exists():
            run([UV, "run", "python", str(SCRIPTS / "gcn_opmix.py"), str(isa_dst), "--top", "20"])
        else:
            warn("no ISA dump to analyze")

    # (5) rocprofv3 kernel-only time + launch resources.
    mojo_kern = None
    if args.no_prof:
        skip("(5) rocprofv3 kernel stats", "--no-prof")
    else:
        section("(5) rocprofv3 kernel stats: mojo v0 (kernel-only + resources)")
        rocprof = shutil.which("rocprofv3")
        if not rocprof:
            warn("rocprofv3 not found — skipping")
        else:
            outdir = Path("/tmp/mb_rocm_mojo")
            shutil.rmtree(outdir, ignore_errors=True)
            run([rocprof, "--kernel-trace", "--output-format", "csv", "-d", str(outdir), "--",
                 str(_RC_MOJO_BIN), "--seq", str(seq), "--head-dim", str(hdim),
                 "--heads", str(heads), "--iters", "3", "--dispatches", "3"], capture=True)
            hits = list(outdir.rglob("*kernel_trace.csv"))
            if hits:
                run([UV, "run", "python", str(SCRIPTS / "rocprof_summary.py"), str(hits[0])])
                mojo_kern = _rc_kernel_only_us(hits[0])
            else:
                warn("rocprof CSV missing — skipping summary")

    # summary + ratio.
    section("summary")
    mojo_wall = mojo.get("min_us") if mojo else None
    mojo_tflops = (mojo.get("gflops_4rcd") / 1000.0) if mojo and mojo.get("gflops_4rcd") else None
    if ref_us is not None:
        print(f"  CK reference (kernel-only): {ref_us:8.1f} us"
              + (f"  ({ref_tflops:.0f} TFLOP/s)" if ref_tflops else ""))
    if mojo_wall is not None:
        print(f"  mojo v0      (wall-clock):  {mojo_wall:8.1f} us")
    if mojo_kern is not None:
        print(f"  mojo v0      (kernel-only): {mojo_kern:8.1f} us")
    mojo_cmp = mojo_kern if mojo_kern is not None else mojo_wall
    ratio = (mojo_cmp / ref_us) if (mojo_cmp and ref_us) else None
    if ratio is not None:
        basis = "kernel-only" if mojo_kern is not None else "wall vs kernel"
        note = "mojo SLOWER" if ratio > 1 else "mojo FASTER"
        print(f"  mojo/CK ratio ({basis}): {ratio:.2f}x  ({note})")
    print(f"  mojo ISA: {isa_dst.relative_to(REPO)}")

    emit_summary({
        "backend": "rocm", "device": platform.node(),
        "shape": f"{batch},{seq},{heads},{hdim}", "dtype": dtype,
        "baseline": "CK-flash (Composable Kernel)",
        "gate": "fail" if Gate.failed else "pass",
        "ck_us": ref_us and round(ref_us, 2),
        "ck_tflops": ref_tflops and round(ref_tflops, 1),
        "mojo_wall_us": mojo_wall and round(mojo_wall, 2),
        "mojo_kernel_us": mojo_kern and round(mojo_kern, 2),
        "mojo_tflops": mojo_tflops and round(mojo_tflops, 1),
        "ratio": ratio and round(ratio, 3),
        "mojo_check_max_error": mojo.get("check_max_error") if mojo else None,
    })
    if Gate.failed:
        print(f"{_RED}ISSUES — one or more gates failed above.{_RST}")
        return 1
    print(f"{_GRN}PASS — mojo v0 correct; CK reference benched (rocm fwd).{_RST}")
    return 0


# ==========================================================================
# CPU backend (pure-PyTorch reference — correctness smoke, no kernel race).
# ==========================================================================

_CP_TIME_SRC = r"""
import sys, time, torch
from flash_attn_mojo.reference import flash_attn_ref
B, S, H, D = (int(x) for x in sys.argv[1:5])
iters = int(sys.argv[5])
name = sys.argv[6]
dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
torch.manual_seed(0)
q = torch.randn(B, S, H, D, dtype=dt)
k = torch.randn(B, S, H, D, dtype=dt)
v = torch.randn(B, S, H, D, dtype=dt)
for _ in range(2):
    flash_attn_ref(q, k, v)
t0 = time.perf_counter()
for _ in range(iters):
    flash_attn_ref(q, k, v)
us = (time.perf_counter() - t0) / iters * 1e6
flops = 4 * B * S * S * H * D
tflops = flops / (us * 1e-6) / 1e12
print(f"RESULT impl=reference kind=fwd shape={B},{S},{H},{D} dtype={name} "
      f"measure=cpu_walltime us={us:.1f} tflops={tflops:.3f}")
"""


def run_cpu(args) -> int:
    # CPU has no flash-attn kernel — the public op falls through to the
    # pure-PyTorch reference. There is nothing to race, so this mode is a
    # correctness smoke-check plus a naive reference wall-clock.
    shape = args.shape if (args.shape and args.shape != CU_CANON) else "2,1024,8,64"
    B, S, H, D = (int(x) for x in shape.split(","))
    dtype = args.dtype if args.dtype in ("fp16", "bf16", "fp32") else "fp32"
    print(f"master_bench: backend=cpu device='{platform.node()}' shape={shape} dtype={dtype}")
    print("  NOTE: no GPU kernel on CPU — flash_attn_func routes to the PyTorch "
          "reference.\n  This is a correctness smoke-check + naive-reference "
          "latency, NOT a perf race.")

    # (a) correctness: tests/test_api.py exercises the real CPU reference path
    # (test_kernels.py is @requires_cuda-skipped on CPU and would verify nothing).
    if args.no_check:
        skip("(a) CPU reference correctness", "--no-check")
    else:
        section("(a) CPU reference correctness (tests/test_api.py)")
        r = run([UV, "run", "python", "-m", "pytest", str(REPO / "tests" / "test_api.py"), "-q"])
        if r.returncode != 0:
            Gate.fail("CPU reference correctness failed (see pytest output above)")
        else:
            print(f"{_GRN}CPU reference correctness OK{_RST}")

    # (b) naive reference latency (informational — no baseline to compare to).
    section("(b) naive reference wall-clock (informational)")
    iters = args.iters if args.iters is not None else 5
    tr = run([UV, "run", "python", "-c", _CP_TIME_SRC,
              str(B), str(S), str(H), str(D), str(iters), dtype], capture=True)
    us = tflops = None
    if tr.returncode == 0:
        for line in (tr.stdout or "").splitlines():
            if line.startswith("RESULT"):
                print("  " + line)
                mu, mt = _RC_RESULT_RE.search(line), _RC_TFLOPS_RE.search(line)
                if mu:
                    us = float(mu.group(1))
                if mt:
                    tflops = float(mt.group(1))
    else:
        warn("reference timing failed (torch unavailable?)")
        if tr.stderr:
            sys.stderr.write(tr.stderr[-500:])
    print("  (SDPA-math reference on CPU — a latency sanity number, not comparable "
          "to the GPU us/TFLOP-s race.)")

    section("summary")
    emit_summary({
        "backend": "cpu", "device": platform.node(), "shape": shape, "dtype": dtype,
        "baseline": None, "note": "no GPU kernel on CPU; reference smoke-check only",
        "gate": "fail" if Gate.failed else "pass",
        "reference_us": us and round(us, 2), "reference_tflops": tflops and round(tflops, 3),
    })
    if Gate.failed:
        print(f"{_RED}ISSUES — CPU correctness gate failed.{_RST}")
        return 1
    print(f"{_GRN}PASS — CPU reference correct (no kernel race on CPU).{_RST}")
    return 0


# ==========================================================================
# Driver.
# ==========================================================================


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("backend")
    g.add_argument("--device", choices=("auto", "cuda", "rocm", "metal", "cpu"),
                   default="auto", help="GPU backend; 'auto' detects (default)")
    g.add_argument("--full", action="store_true", help="multi-shape sweep")
    g.add_argument("-v", "--verbose", action="store_true")

    g = p.add_argument_group("shape (cuda / cpu use B,S,H,D; metal / rocm use --seq etc.)")
    g.add_argument("--shape", default=None, help=f"[cuda/cpu] B,S,H,D (cuda default {CU_CANON})")
    g.add_argument("--seq", type=int, default=None, help="[metal/rocm] sequence length")
    g.add_argument("--head-dim", type=int, default=None, help="[metal/rocm] head dim")
    g.add_argument("--heads", type=int, default=None, help="[metal/rocm] number of heads")
    g.add_argument("--batch", type=int, default=None, help="[rocm] batch size")
    g.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=None,
                   help="cuda default bf16; metal/rocm default fp16; cpu default fp32")

    g = p.add_argument_group("envelope [cuda only]")
    g.add_argument("--kind", choices=("fwd", "bwd"), default="fwd", help="[cuda] fwd/bwd")
    g.add_argument("--causal", action="store_true", help="[cuda]")
    g.add_argument("--hkv", type=int, default=0, help="[cuda] KV heads (GQA); 0=MHA")
    g.add_argument("--varlen", action="store_true", help="[cuda] packed varlen")
    g.add_argument("--window", type=int, default=0, help="[cuda] sliding-window left")
    g.add_argument("--softcap", type=float, default=0.0, help="[cuda] logit softcap")

    g = p.add_argument_group("iterations")
    g.add_argument("--iters", type=int, default=None, help="timed iters (backend default)")
    g.add_argument("--runs", type=int, default=3, help="[cuda] timed runs/impl for spread")
    g.add_argument("--rounds", type=int, default=None, help="[metal] interleave rounds")
    g.add_argument("--dispatches", type=int, default=5, help="[metal/rocm] dispatches/cb")

    g = p.add_argument_group("phase toggles")
    g.add_argument("--no-check", action="store_true", help="skip correctness")
    g.add_argument("--no-prof", action="store_true", help="skip deep profiler (ncu/rocprofv3)")
    g.add_argument("--no-ncu", action="store_true", help="[cuda] alias for --no-prof")
    g.add_argument("--no-asm", action="store_true", help="skip PTX/AIR/GCN op-mix")
    g.add_argument("--no-lock", action="store_true", help="[cuda/metal] don't lock clocks")
    g.add_argument("--no-clean", action="store_true", help="[cuda] keep the JIT cache")
    g.add_argument("--no-build", action="store_true", help="[metal/rocm] skip binary rebuild")
    g.add_argument("--no-walltime", action="store_true", help="[cuda] skip wall-clock phase")
    g.add_argument("--no-kernel-time", action="store_true",
                   help="[metal] skip the xctrace mojo-kernel-time phase "
                        "(default on; the mojo column stays wall-clock)")

    g = p.add_argument_group("gating")
    g.add_argument("--no-gate", action="store_true", help="[cuda] report a SLOWER shape, don't fail")
    g.add_argument("--gate", action="store_true", help="[metal] fail on a mojo ratchet regression")
    g.add_argument("--refresh-baseline", action="store_true", help="[metal] reseed the ratchet")
    g.add_argument("--refresh-fa4-ptx", action="store_true", help="[cuda]")

    g = p.add_argument_group("metal-specific")
    g.add_argument("--impls", default=None, help="[metal] lanes (default mojo,mfa,ccv)")
    g.add_argument("--profile", metavar="IMPL", default=None, help="[metal] xctrace-record one impl")
    g.add_argument("--jsonl", type=Path, default=None, help="[metal] append raw per-run JSON")

    g = p.add_argument_group("misc")
    g.add_argument("--quick", action="store_true", help="[rocm] small shape, fast")

    args = p.parse_args()

    global VERBOSE
    VERBOSE = args.verbose
    os.chdir(REPO)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    # Unified aliases.
    if args.no_ncu:
        args.no_prof = True

    backend = detect_backend(args.device)

    # Cross-backend envelope guards.
    if args.kind == "bwd" and backend != "cuda":
        print(f"{_RED}[FAIL]{_RST} --kind bwd is CUDA-only "
              f"(metal/rocm/cpu are v0 forward-only). Detected backend: {backend}.")
        return 1

    # cuda default shape/dtype.
    if backend == "cuda":
        if args.shape is None:
            args.shape = CU_CANON
        if args.dtype is None:
            args.dtype = "bf16"

    dispatch = {"cuda": run_cuda, "rocm": run_rocm, "metal": run_metal, "cpu": run_cpu}[backend]
    return dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
