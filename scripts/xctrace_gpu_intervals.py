#!/usr/bin/env python3
"""Extract per-encoder GPU execution intervals from an xctrace trace.

Usage:
  python scripts/xctrace_gpu_intervals.py trace.trace [label-substring]

Exports the `metal-gpu-intervals` table (the scriptable subset of
Instruments' Metal System Trace — per-encoder GPU wall times) and
prints each matching interval plus a summary. GPU hardware counters
(ALU busy, bandwidth) are NOT in headless traces; that part of
Instruments is GUI-only on Apple silicon.
"""

import subprocess
import sys
import xml.etree.ElementTree as ET


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    trace = sys.argv[1]
    needle = sys.argv[2] if len(sys.argv) > 2 else ""

    out = subprocess.run(
        [
            "xctrace", "export", "--input", trace,
            "--xpath",
            '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]',
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.exit(f"xctrace export failed:\n{out.stderr}")

    root = ET.fromstring(out.stdout)
    durations_ns = []
    for row in root.findall(".//row"):
        label = row.find(".//formatted-label")
        fmt = label.get("fmt", "") if label is not None else ""
        if needle and needle not in fmt:
            continue
        dur = row.find(".//duration")
        if dur is None:
            continue
        durations_ns.append(int(dur.text))
        start = row.find(".//start-time")
        print(f"{start.get('fmt') if start is not None else '?':>14}  "
              f"{dur.get('fmt'):>12}  {fmt}")
    if not durations_ns:
        sys.exit(f"no GPU intervals matched {needle!r}")
    total = sum(durations_ns)
    print(f"\n{len(durations_ns)} encoder intervals, total GPU time "
          f"{total / 1e6:.3f} ms, mean {total / len(durations_ns) / 1e6:.3f} ms")


if __name__ == "__main__":
    main()
