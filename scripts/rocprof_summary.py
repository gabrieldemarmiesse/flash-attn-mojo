#!/usr/bin/env python3
"""Summarize rocprofv3 --kernel-trace CSVs — the ncu-stats analog.

rocprofv3 emits one row per kernel dispatch with the wall-accurate
device duration (Start/End_Timestamp, ns) plus the launch resource
footprint (VGPR_Count, Scratch_Size, LDS_Block_Size, grid/block dims).
This groups those rows by kernel name and prints, per kernel, the
dispatch count, mean/min device time, and resources — so the Mojo
kernel and the torch reference can be compared kernel-only and
apples-to-apples (both under the same profiler), the way ncu_compare.py
puts the two NVIDIA kernels side by side.

The first dispatch of each kernel is dropped as warmup unless --keep-first.

Usage:
    uv run python scripts/rocprof_summary.py mojo_kernel_trace.csv
    uv run python scripts/rocprof_summary.py mojo.csv ref.csv   # both
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    """kernel-name -> list of (dur_us, vgpr, scratch, lds, grid, block)."""
    by_name = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Kind") and "KERNEL" not in r["Kind"].upper():
                continue
            try:
                dur = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
            except (KeyError, ValueError):
                continue
            grid = "x".join(
                r.get(f"Grid_Size_{a}", "?") for a in ("X", "Y", "Z")
            )
            block = "x".join(
                r.get(f"Workgroup_Size_{a}", "?") for a in ("X", "Y", "Z")
            )
            by_name[r.get("Kernel_Name", "?")].append(
                (
                    dur,
                    r.get("VGPR_Count", "-"),
                    r.get("Scratch_Size", "-"),
                    r.get("LDS_Block_Size", "-"),
                    grid,
                    block,
                )
            )
    return by_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument(
        "--keep-first", action="store_true",
        help="do not drop each kernel's first dispatch as warmup",
    )
    ap.add_argument(
        "--name-width", type=int, default=44, help="kernel-name column width"
    )
    args = ap.parse_args()

    rows = []
    for path in args.csv:
        for name, samples in load(path).items():
            if not args.keep_first and len(samples) > 1:
                samples = samples[1:]
            durs = [s[0] for s in samples]
            mean = sum(durs) / len(durs)
            vgpr, scratch, lds, grid, block = samples[-1][1:]
            rows.append(
                (
                    name[: args.name_width],
                    len(durs),
                    mean,
                    min(durs),
                    vgpr,
                    scratch,
                    lds,
                    grid,
                    block,
                )
            )
    rows.sort(key=lambda r: -r[2])

    headers = (
        "kernel", "n", "mean_us", "min_us", "vgpr", "scratch", "lds",
        "grid", "block",
    )
    cells = [
        (n, str(c), f"{mu:.1f}", f"{mn:.1f}", str(v), str(sc), str(l), g, b)
        for (n, c, mu, mn, v, sc, l, g, b) in rows
    ]
    widths = [
        max(len(str(x[i])) for x in cells + [headers])
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print(fmt.format(*c))


if __name__ == "__main__":
    main()
