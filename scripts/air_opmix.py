#!/usr/bin/env python3
"""AIR op-mix histogram / diff — the Metal analog of the SASS op-mix diff.

Usage:
  python scripts/air_opmix.py kernel.air                 # histogram
  python scripts/air_opmix.py a.air b.air                # diff (b - a)
  python scripts/air_opmix.py --function attention a.air b.air

Accepts .air (LLVM bitcode — disassembled via `xcrun metal-objdump -d`)
or already-textual .ll files. Counts instruction opcodes inside
`define` bodies; `call` instructions are counted by callee (so
`call air.simdgroup_matrix_8x8_multiply_accumulate.*` shows up as its
own row — the HGMMA-count equivalent). Vector ops are suffixed with
their element count (`fmul.v2`) since AIR leans on <2 x half> math.

Same-MMA-count + diff of the rest = same tensor work, different
overhead — exactly how the NVIDIA-side op-mix diff was read.
"""

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# %x = OPCODE ... | OPCODE ... (store/br/ret have no result binding)
_INST = re.compile(r"^\s*(?:%[\w.\-]+\s*=\s*)?(?:tail\s+)?(\w+)")
_CALLEE = re.compile(r"call[^@]*@([\w.\$\-]+)")
_VEC = re.compile(r"<(\d+) x (?:half|float|i\d+|bfloat)>")
_DEFINE = re.compile(r"^define\b.*@([\w.\$\-]+)\(")

# Flow/metadata noise that never decides a perf diff.
_SKIP = {"unreachable"}


def disassemble(path: Path) -> str:
    if path.suffix == ".ll" or path.suffix == ".txt":
        return path.read_text()
    out = subprocess.run(
        ["xcrun", "metal-objdump", "-d", str(path)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.exit(f"metal-objdump failed on {path}:\n{out.stderr}")
    return out.stdout


def opmix(text: str, function: str | None) -> tuple[Counter, list[str]]:
    counts: Counter = Counter()
    functions: list[str] = []
    in_body = False
    wanted = False
    for line in text.splitlines():
        if not in_body:
            m = _DEFINE.match(line)
            if m and line.rstrip().endswith("{"):
                functions.append(m.group(1))
                in_body = True
                wanted = function is None or m.group(1) == function
            continue
        if line.startswith("}"):
            in_body = False
            continue
        if not wanted:
            continue
        m = _INST.match(line)
        if not m:
            continue
        op = m.group(1)
        if op in _SKIP or not op.islower():
            continue
        if op in ("call", "invoke", "musttail"):
            c = _CALLEE.search(line)
            op = f"call {c.group(1)}" if c else "call <indirect>"
            if op.startswith("call llvm.lifetime"):
                continue
        else:
            v = _VEC.search(line)
            if v:
                op = f"{op}.v{v.group(1)}"
        counts[op] += 1
    return counts, functions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path, help="1 or 2 .air/.ll files")
    ap.add_argument("--function", help="restrict to one kernel function")
    ap.add_argument("--top", type=int, default=0, help="show only top-N rows")
    args = ap.parse_args()
    if len(args.files) > 2:
        ap.error("expected 1 or 2 files")

    mixes = []
    for f in args.files:
        counts, fns = opmix(disassemble(f), args.function)
        if not counts:
            hint = f" (functions present: {', '.join(fns) or 'none'})"
            sys.exit(f"no instructions counted in {f}{hint}")
        mixes.append(counts)

    if len(mixes) == 1:
        (counts,) = mixes
        rows = counts.most_common(args.top or None)
        width = max(len(op) for op, _ in rows)
        for op, n in rows:
            print(f"{op:<{width}}  {n}")
        print(f"{'TOTAL':<{width}}  {sum(counts.values())}")
        return

    a, b = mixes
    ops = sorted(set(a) | set(b), key=lambda o: (-abs(b[o] - a[o]), -(a[o] + b[o]), o))
    if args.top:
        ops = ops[: args.top]
    width = max(len(o) for o in ops)
    na, nb = args.files[0].name, args.files[1].name
    print(f"{'op':<{width}}  {na:>12} {nb:>12} {'delta':>8}")
    for op in ops:
        d = b[op] - a[op]
        print(f"{op:<{width}}  {a[op]:>12} {b[op]:>12} {d:>+8}")
    print(
        f"{'TOTAL':<{width}}  {sum(a.values()):>12} {sum(b.values()):>12}"
        f" {sum(b.values()) - sum(a.values()):>+8}"
    )


if __name__ == "__main__":
    main()
