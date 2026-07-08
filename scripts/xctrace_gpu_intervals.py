#!/usr/bin/env python3
"""Per-encoder GPU intervals from an xctrace 'Metal System Trace'.

The scriptable subset of Instruments' Metal System Trace: exports the
`metal-gpu-intervals` table (per-encoder GPU wall times), the
`gpu-performance-state-intervals` table (the DVFS clock timeline), and the
device-global `metal-gpu-state-intervals` duty cycle, then groups the
kernel's encoders by (channel, label, clock) and prints median/min/max per
group + the Active/Idle duty cycle. GPU *hardware* counters (ALU busy,
bandwidth, occupancy) are NOT in headless traces — that part of Instruments
is GUI-only on Apple silicon.

Robustness the naive version lacked (both are why the old one crashed):
  * the export uses a global id/ref value-dictionary — the first use of a
    value carries `id=` + data, later uses are `<tag ref=.../>` empty refs;
    resolve them or `int(<duration/>)` throws on the repeats.
  * each row has TWO `duration` children (GPU time, then CPU->GPU latency) —
    the GPU interval is the FIRST.
  * `metal-gpu-intervals` lumps every process's GPU work together
    (WindowServer compositing dominates the row count) — filter by process.

Usage:
  xctrace_gpu_intervals.py trace.trace [--process NAME] [--label SUBSTR]

Exit status: nonzero if the intervals table can't be exported (an
unfinalized/corrupt trace — xctrace crashes on finalize intermittently, so
the caller should re-record), else zero.
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

_INTERVALS = "metal-gpu-intervals"
_CLOCK = "gpu-performance-state-intervals"
_STATE = "metal-gpu-state-intervals"
_CLOCK_RANK = {"Minimum": 0, "Low": 1, "Medium": 2, "High": 3, "Maximum": 4}


def _export(trace: str, schema: str) -> bytes:
    out = subprocess.run(
        ["xctrace", "export", "--input", trace, "--xpath",
         f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'],
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="replace").strip())
    return out.stdout


def _resolver(root):
    registry = {el.get("id"): el for el in root.iter() if el.get("id")}

    def resolve(el):
        ref = el.get("ref")
        return registry.get(ref, el) if ref else el

    return resolve


def _clock_timeline(trace: str) -> list[tuple[int, int, str]]:
    try:
        root = ET.fromstring(_export(trace, _CLOCK))
    except RuntimeError:
        return []
    resolve = _resolver(root)
    windows = []
    for row in root.iter("row"):
        start = dur = state = None
        for k in row:
            if k.tag == "start-time":
                start = int(resolve(k).text)
            elif k.tag == "duration":
                dur = int(resolve(k).text)
            elif k.tag == "gpu-performance-state":
                state = resolve(k).get("fmt", "")
        if start is not None and state:
            windows.append((start, start + (dur or 0), state))
    windows.sort()
    return windows


def _clock_at(windows, t: int) -> str:
    for s, e, st in windows:
        if s <= t < e:
            return st
    return ""


def _duty(trace: str) -> dict | None:
    try:
        root = ET.fromstring(_export(trace, _STATE))
    except RuntimeError:
        return None
    resolve = _resolver(root)
    by_state: dict[str, int] = defaultdict(int)
    for row in root.iter("row"):
        state, dur = None, 0
        for k in row:
            if k.tag == "gpu-state":
                state = resolve(k).get("fmt", "")
            elif k.tag == "duration":
                txt = resolve(k).text
                dur = int(txt) if txt else 0
        if state:
            by_state[state] += dur
    total = sum(by_state.values())
    if not total:
        return None
    active = by_state.get("Active", 0)
    return {"active_ms": active / 1e6, "idle_ms": by_state.get("Idle", 0) / 1e6,
            "busy_pct": active / total * 100.0}


def _normalize_label(lbl: str) -> str:
    lbl = re.sub(r"Command Buffer \d+:", "", lbl)
    lbl = re.sub(r" \d+$", "", lbl)
    return lbl.strip() or "?"


def steady_state_kernel_us(trace: str, process: str) -> float | None:
    """True per-dispatch GPU kernel time (us), fragmentation-proof.

    Matches the reference CLIs' protocol (min over dispatches of the
    command-buffer gpuEnd-gpuStart) from the intervals table, robust
    to the two ways that table lies:

    1. FRAGMENTATION. A long dispatch is split into several intervals
       when the DVFS clock transitions mid-dispatch (each interval is
       a fraction of the real dispatch), so a naive median/min
       under-reports long kernels. Fragmentation only ever *splits* a
       dispatch — never merges two, since each dispatch is its own
       compute encoder — so full un-fragmented dispatches form the
       TOP cluster of the distribution and every fragment sits below
       it. We keep only intervals >= 0.6x the max, discarding
       fragments.
    2. CONTENTION. Within the full-dispatch cluster, some dispatches
       run long (WindowServer/other-process GPU contention, DVFS
       jitter). The reference rejects these by taking the MIN over
       dispatches; we take the min of the surviving cluster.

    The clock used is whichever the GPU actually settled at — short
    kernels never ramp past Medium (no clock locking exists on
    macOS), which is fine: the references run at the same ambient
    clock, so the comparison stays fair.

    Returns None if no compute intervals are found. Raises
    RuntimeError if the trace can't be exported (re-record).
    """
    xml = _export(trace, _INTERVALS)
    windows = _clock_timeline(trace)
    by_clock: dict[str, list[float]] = defaultdict(list)
    for channel, raw_label, start_ns, ns in _parse_intervals(
        xml, process, normalize=False
    ):
        if "compute" not in channel.lower() or "blit" in raw_label.lower():
            continue
        by_clock[_clock_at(windows, start_ns) or "?"].append(ns / 1e3)
    if not by_clock:
        return None
    # Bucket by DVFS clock FIRST: a dispatch's duration depends on the
    # clock it ran at, so mixing clocks (warmup ramp vs steady state)
    # would compare apples to oranges. Take the highest clock the GPU
    # settled at — that's where the references run too.
    top = max(by_clock, key=lambda c: _CLOCK_RANK.get(c, -1))
    durs = by_clock[top]
    hi = max(durs)
    below = [d for d in durs if d < 0.85 * hi]  # candidate fragments
    if not below or min(below) >= 0.5 * hi:
        # No fragmentation within this clock: every interval is a whole
        # dispatch (the kernel fits inside one clock window). Best-case
        # per the reference protocol = the min.
        return min(durs)
    # Fragmentation present: clean un-fragmented dispatches form the
    # tight top cluster (>= 0.85x max); fragments sit well below.
    # Median of the cluster — robust to a lone lopsided split.
    return statistics.median([d for d in durs if d >= 0.85 * hi])


def _parse_intervals(xml: bytes, process: str, normalize: bool = True):
    root = ET.fromstring(xml)
    resolve = _resolver(root)
    for row in root.iter("row"):
        kids = list(row)
        proc, channel, flabel = "", "", None
        durations = []
        for k in kids:
            if k.tag == "process" and not proc:
                proc = resolve(k).get("fmt", "")
            elif k.tag == "duration":
                durations.append(resolve(k))
            elif k.tag == "gpu-channel-name" and not channel:
                channel = resolve(k).get("fmt", "")
            elif k.tag == "formatted-label" and flabel is None:
                flabel = resolve(k)
        if process and process not in proc:
            continue
        if not durations or durations[0].text is None:
            continue
        label = "?"
        if flabel is not None:
            s = flabel.find("string")
            if s is not None:
                label = resolve(s).get("fmt", "") or "?"
        start_ns = 0
        for k in kids:
            if k.tag == "start-time":
                start_ns = int(resolve(k).text)
                break
        out_label = _normalize_label(label) if normalize else label
        yield channel, out_label, start_ns, int(durations[0].text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--process", default="",
                    help="only rows whose process name contains this "
                         "(e.g. bench_mojo_metal); default: all processes")
    ap.add_argument("--label", default="", help="only encoder labels containing this")
    ap.add_argument("--kernel-us", action="store_true",
                    help="machine mode: print ONLY the steady-state per-dispatch "
                         "GPU kernel time in us (fragmentation-proof; see "
                         "steady_state_kernel_us) and nothing else")
    args = ap.parse_args()

    if args.kernel_us:
        try:
            us = steady_state_kernel_us(args.trace, args.process)
        except RuntimeError as e:
            sys.exit(f"xctrace export failed (unfinalized trace? re-record): {e}")
        if us is None:
            sys.exit("no compute intervals")
        print(f"{us:.2f}")
        return

    try:
        xml = _export(args.trace, _INTERVALS)
    except RuntimeError as e:
        sys.exit(f"xctrace export failed (unfinalized trace? re-record): {e}")

    windows = _clock_timeline(args.trace)
    bucket: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for channel, label, start_ns, ns in _parse_intervals(xml, args.process):
        if args.label and args.label not in label:
            continue
        clock = _clock_at(windows, start_ns) or "?"
        bucket[(channel, label, clock)].append(ns)

    if not bucket:
        who = f" for process ~{args.process!r}" if args.process else ""
        sys.exit(f"no GPU encoder intervals{who} in {args.trace}")

    groups = []
    for (channel, label, clock), durs in bucket.items():
        groups.append({
            "channel": channel, "label": label, "clock": clock, "count": len(durs),
            "median_us": statistics.median(durs) / 1e3,
            "min_us": min(durs) / 1e3, "max_us": max(durs) / 1e3,
        })
    groups.sort(key=lambda g: -sum([g["median_us"] * g["count"]]))

    hdr = (f"    {'channel':>10} | {'clock':>8} | {'count':>5} | {'median':>9} | "
           f"{'min':>9} | {'max':>9} | encoder")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for g in groups:
        print(f"    {g['channel']:>10} | {g['clock']:>8} | {g['count']:>5} | "
              f"{g['median_us']:8.2f}u | {g['min_us']:8.2f}u | {g['max_us']:8.2f}u | "
              f"{g['label']}")

    # Headline: the kernel's compute-command encoder at the highest clock seen.
    kernel = [g for g in groups if "compute command" in g["label"].lower()] or [
        g for g in groups
        if "compute" in g["channel"].lower() and "blit" not in g["label"].lower()]
    if kernel:
        hl = max(kernel, key=lambda g: (_CLOCK_RANK.get(g["clock"], -1), g["count"]))
        print(f"\n    headline kernel encoder: {hl['label']} @ {hl['clock']} clock — "
              f"median {hl['median_us']:.2f} us over {hl['count']} intervals")

    d = _duty(args.trace)
    if d:
        print(f"    GPU duty cycle (device-global): {d['busy_pct']:.1f}% active "
              f"({d['active_ms']:.2f} ms active / {d['idle_ms']:.2f} ms idle)")
        if d["busy_pct"] < 40.0:
            print("      -> low residency: launch/sync-bound, not compute-bound.")


if __name__ == "__main__":
    main()
