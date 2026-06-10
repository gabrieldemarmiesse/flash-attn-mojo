"""PTX instruction-mix statistics.

Parses one or two .ptx files and prints per-opcode instruction
counts (and register declarations). With two files, prints them
side-by-side so the Mojo kernel's op mix can be diffed against the
FA4 reference at a glance — wrong-instruction smells (e.g. scalar
`ld.global` where the reference uses `cp.async.bulk.tensor`, or
f32 `mul`/`add` chains where the reference has `fma`) jump out.

Usage:
    uv run python scripts/ptx_stats.py reference_ptx/fa4_*.ptx ptx/mojo_fwd_fa4.ptx
    uv run python scripts/ptx_stats.py ptx/mojo_fwd_fa4.ptx
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Opcode → coarse class. First matching prefix wins (ordered).
_CLASSES = [
    ("wgmma", "tensor-core"),
    ("mma", "tensor-core"),
    ("cp.async.bulk.tensor", "tma"),
    ("cp.async.bulk", "tma"),
    ("cp.async", "cp.async"),
    ("cp.reduce", "tma"),
    ("mbarrier", "barrier"),
    ("bar.", "barrier"),
    ("barrier", "barrier"),
    ("fence", "barrier"),
    ("membar", "barrier"),
    ("ld.global", "gmem"),
    ("st.global", "gmem"),
    ("ld.shared", "smem"),
    ("st.shared", "smem"),
    ("ldmatrix", "smem"),
    ("stmatrix", "smem"),
    ("ld.param", "param"),
    ("ld.const", "param"),
    ("ex2", "sfu"),
    ("lg2", "sfu"),
    ("rcp", "sfu"),
    ("sqrt", "sfu"),
    ("fma", "fp-alu"),
    ("mul.f", "fp-alu"),
    ("add.f", "fp-alu"),
    ("sub.f", "fp-alu"),
    ("max.f", "fp-alu"),
    ("min.f", "fp-alu"),
    ("mul.rn.f", "fp-alu"),
    ("cvt", "cvt"),
    ("mov", "mov"),
    ("shfl", "shuffle"),
    ("selp", "int-alu"),
    ("setp", "int-alu"),
    ("add.s", "int-alu"),
    ("add.u", "int-alu"),
    ("sub.s", "int-alu"),
    ("mul.lo", "int-alu"),
    ("mad.lo", "int-alu"),
    ("mad.wide", "int-alu"),
    ("shl", "int-alu"),
    ("shr", "int-alu"),
    ("and", "int-alu"),
    ("or", "int-alu"),
    ("xor", "int-alu"),
    ("not", "int-alu"),
    ("min.s", "int-alu"),
    ("max.s", "int-alu"),
    ("bfe", "int-alu"),
    ("bfi", "int-alu"),
    ("prmt", "int-alu"),
    ("lop3", "int-alu"),
    ("bra", "control"),
    ("ret", "control"),
    ("call", "control"),
    ("elect", "control"),
    ("vote", "control"),
    ("activemask", "control"),
    ("griddepcontrol", "control"),
    ("red.", "atomic"),
    ("atom", "atomic"),
    ("prefetch", "gmem"),
]


def classify(op: str) -> str:
    for prefix, cls in _CLASSES:
        if op.startswith(prefix):
            return cls
    return "other"


_LABEL_RE = re.compile(r"^\$?[A-Za-z_$][\w$]*:")
_REG_RE = re.compile(r"^\.reg\s+\.(\w+)\s+%[\w$]+<(\d+)>")


def parse_ptx(path: Path) -> tuple[Counter, Counter]:
    """Return (opcode counter, register-declaration counter)."""
    ops: Counter = Counter()
    regs: Counter = Counter()
    text = path.read_text(errors="replace")
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        # Strip predicate guard.
        if line.startswith("@"):
            line = line.split(None, 1)[1] if " " in line else ""
            if not line:
                continue
        if _LABEL_RE.match(line):
            rest = line.split(":", 1)[1].strip()
            if not rest:
                continue
            line = rest
        if line.startswith("."):
            m = _REG_RE.match(line)
            if m:
                regs[m.group(1)] += int(m.group(2))
            continue
        if line in ("{", "}") or line.startswith(("{", "}")):
            continue
        op = line.split()[0].rstrip(";")
        if not op:
            continue
        ops[op] += 1
    return ops, regs


def fmt_table(rows: list[tuple], headers: tuple) -> str:
    widths = [
        max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))
    ]
    out = []
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    out.append(line)
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ptx", nargs="+", type=Path, help="one or two .ptx files")
    p.add_argument("--top", type=int, default=45, help="rows in opcode table")
    args = p.parse_args()
    if len(args.ptx) > 2:
        p.error("at most two ptx files")

    parsed = [parse_ptx(f) for f in args.ptx]
    names = [f.name[:28] for f in args.ptx]

    # ---- per-class summary ----
    print("== instruction classes ==")
    class_counts = []
    for ops, _ in parsed:
        c: Counter = Counter()
        for op, n in ops.items():
            c[classify(op)] += n
        class_counts.append(c)
    all_classes = sorted(
        {k for c in class_counts for k in c},
        key=lambda k: -max(c.get(k, 0) for c in class_counts),
    )
    rows = [
        tuple([cls] + [c.get(cls, 0) for c in class_counts])
        for cls in all_classes
    ]
    rows.append(tuple(["TOTAL"] + [sum(c.values()) for c in class_counts]))
    print(fmt_table(rows, tuple(["class"] + names)))

    # ---- per-opcode table ----
    print("\n== opcodes ==")
    all_ops = {k for ops, _ in parsed for k in ops}
    ranked = sorted(
        all_ops, key=lambda k: -max(ops.get(k, 0) for ops, _ in parsed)
    )
    rows = [
        tuple([op] + [ops.get(op, 0) for ops, _ in parsed])
        for op in ranked[: args.top]
    ]
    dropped = len(ranked) - args.top
    print(fmt_table(rows, tuple(["opcode"] + names)))
    if dropped > 0:
        print(f"... {dropped} more opcodes (raise --top to see)")

    # ---- register declarations ----
    print("\n== .reg declarations ==")
    all_regs = {k for _, regs in parsed for k in regs}
    rows = [
        tuple([rt] + [regs.get(rt, 0) for _, regs in parsed])
        for rt in sorted(
            all_regs, key=lambda k: -max(regs.get(k, 0) for _, regs in parsed)
        )
    ]
    print(fmt_table(rows, tuple(["reg type"] + names)))


if __name__ == "__main__":
    main()
