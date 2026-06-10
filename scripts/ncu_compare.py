"""Side-by-side headline metrics from two ncu .ncu-rep reports.

Imports each report with `ncu --import --csv --page details` and
prints the key rows (duration, SOL compute/memory, occupancy,
registers, launch config) in one table: FA4 vs Mojo.

Usage (ncu binary path given by the caller, see master_bench.sh):
    python scripts/ncu_compare.py --ncu "<ncu cmd>" fa4.ncu-rep mojo.ncu-rep
"""

from __future__ import annotations

import argparse
import csv
import io
import shlex
import subprocess
import sys

_WANTED = [
    "Duration",
    "Compute (SM) Throughput",
    "Memory Throughput",
    "DRAM Throughput",
    "Elapsed Cycles",
    "SM Active Cycles",
    "Achieved Occupancy",
    "Theoretical Occupancy",
    "Registers Per Thread",
    "Static Shared Memory Per Block",
    "Dynamic Shared Memory Per Block",
    "Threads",
    "Grid Size",
    "Block Size",
    "Waves Per SM",
    "Achieved Active Warps Per SM",
]


def read_report(ncu_cmd: list[str], path: str) -> dict[str, str]:
    out = subprocess.run(
        ncu_cmd + ["--import", path, "--csv", "--page", "details"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    metrics: dict[str, str] = {}
    reader = csv.DictReader(io.StringIO(out))
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        if name in _WANTED and name not in metrics:
            val = (row.get("Metric Value") or "").strip()
            unit = (row.get("Metric Unit") or "").strip()
            metrics[name] = f"{val} {unit}".strip()
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ncu", required=True, help="ncu command (shell-quoted)")
    p.add_argument("reports", nargs=2, metavar=("FA4_REP", "MOJO_REP"))
    p.add_argument("--labels", default="fa4,mojo")
    args = p.parse_args()

    ncu_cmd = shlex.split(args.ncu)
    labels = args.labels.split(",")
    data = []
    for path in args.reports:
        try:
            data.append(read_report(ncu_cmd, path))
        except subprocess.CalledProcessError as e:
            print(f"failed to import {path}: {e.stderr[-500:]}", file=sys.stderr)
            sys.exit(1)

    w0 = max(len(m) for m in _WANTED)
    w1 = max([len(data[0].get(m, "-")) for m in _WANTED] + [len(labels[0])])
    print(f"{'metric'.ljust(w0)}  {labels[0].ljust(w1)}  {labels[1]}")
    print(f"{'-' * w0}  {'-' * w1}  {'-' * 10}")
    for m in _WANTED:
        a = data[0].get(m, "-")
        b = data[1].get(m, "-")
        if a == "-" and b == "-":
            continue
        print(f"{m.ljust(w0)}  {a.ljust(w1)}  {b}")


if __name__ == "__main__":
    main()
