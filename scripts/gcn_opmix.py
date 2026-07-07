#!/usr/bin/env python3
"""AMDGCN ISA instruction-mix statistics — the AMD analog of ptx_stats.py.

Parses one or two AMDGCN assembly files (`.s`, as emitted by Mojo's
`dump_asm` on a HIP target — see bench/bench_mojo_rocm.mojo) and prints
per-opcode instruction counts, a coarse instruction-class summary, and
the kernel's register/LDS/scratch resource footprint. With two files,
prints them side-by-side so the Mojo kernel's op mix can be diffed at a
glance — wrong-instruction smells jump out (e.g. no `v_mfma_*` where a
matrix-core kernel should have them, `v_pk_fma_f32` FMA chains where a
tensor kernel belongs, or a `v_*` spill storm to scratch).

The direct read is the same as the NVIDIA-side PTX/SASS op-mix diff:
same `v_mfma_*` count == same tensor-core work; diff the rest to see the
overhead delta. On this v0 (no matrix cores) the interesting canaries
are the vgpr_spill_count (must be 0 on a tuned kernel), the LDS size,
and the ds_bpermute/ds_read (shuffle + LDS) traffic.

Usage:
    uv run python scripts/gcn_opmix.py asm/mojo_fwd_rocm_d128.s
    uv run python scripts/gcn_opmix.py ref.s asm/mojo_fwd_rocm_d128.s
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def classify(op: str) -> str:
    """Coarse instruction class. First matching rule wins."""
    # Matrix / dot cores — the tensor-core equivalent.
    if op.startswith(("v_mfma", "v_smfmac", "v_wmma")):
        return "matrix"
    if op.startswith("v_dot"):
        return "dot"
    # Cross-lane data movement (the shuffle/shfl equivalent).
    if op.startswith(
        ("ds_bpermute", "ds_permute", "ds_swizzle", "v_permlane")
    ):
        return "shuffle"
    # LDS (shared memory).
    if op.startswith("ds_"):
        return "lds"
    # Global / scratch (spill!) memory.
    if op.startswith(("scratch_load", "scratch_store")):
        return "scratch"
    if op.startswith(("global_load", "buffer_load", "flat_load")):
        return "gmem-ld"
    if op.startswith(("global_store", "buffer_store", "flat_store")):
        return "gmem-st"
    if op.startswith(("global_atomic", "buffer_atomic", "flat_atomic")):
        return "atomic"
    # Scalar (uniform / constant) memory loads.
    if op.startswith(("s_load", "s_buffer_load")):
        return "smem-ld"
    # Special-function unit (transcendentals).
    if op.startswith(
        ("v_exp", "v_log", "v_rcp", "v_rsq", "v_sqrt", "v_sin", "v_cos")
    ):
        return "sfu"
    if op.startswith("v_cvt"):
        return "cvt"
    if op.startswith(("v_mov", "s_mov")):
        return "mov"
    if op.startswith(("v_cmp", "s_cmp")):
        return "cmp"
    if op.startswith(("v_cndmask", "s_cselect")):
        return "select"
    # Waits / barriers / cache ops (the mbarrier/bar.sync equivalent).
    if op.startswith(
        ("s_waitcnt", "s_barrier", "buffer_wbinvl", "buffer_inv",
         "s_wakeup", "buffer_wbl2", "s_dcache")
    ):
        return "sync"
    # Control flow.
    if op.startswith(
        ("s_branch", "s_cbranch", "s_setpc", "s_swappc", "s_call",
         "s_endpgm", "s_getpc", "s_and_saveexec", "s_or_saveexec",
         "s_xor_saveexec", "s_andn2_saveexec")
    ):
        return "control"
    if op in ("s_nop", "v_nop"):
        return "nop"
    # Arithmetic — split fp vs int by dtype suffix.
    is_fp = ("f32" in op) or ("f16" in op) or ("f64" in op)
    if op.startswith("v_"):
        return "fp-alu" if is_fp else "int-alu"
    if op.startswith("s_"):
        return "scalar-alu"
    return "other"


# An instruction line is indented and begins with a `v_`/`s_`/`ds_`/…
# mnemonic. Directives (`.`), labels (`…:`), comments (`;`,`//`) skipped.
_MNEMONIC = re.compile(
    r"^\s+((?:v|s|ds|global|buffer|flat|scratch)_[a-z0-9_]+)\b"
)

# Resource footprint, straight from the emitted .amdhsa / .kd metadata.
_RES = {
    "vgpr_count": re.compile(r"^\s*\.vgpr_count:\s*(\d+)"),
    "vgpr_spill": re.compile(r"^\s*\.vgpr_spill_count:\s*(\d+)"),
    "sgpr_count": re.compile(r"^\s*\.sgpr_count:\s*(\d+)"),
    "sgpr_spill": re.compile(r"^\s*\.sgpr_spill_count:\s*(\d+)"),
    "lds_bytes": re.compile(r"\.amdhsa_group_segment_fixed_size\s+(\d+)"),
    "scratch_bytes": re.compile(
        r"\.amdhsa_private_segment_fixed_size\s+(\d+)"
    ),
}


def parse(path: Path) -> tuple[Counter, dict[str, int]]:
    ops: Counter = Counter()
    res: dict[str, int] = {}
    for raw in path.read_text(errors="replace").splitlines():
        m = _MNEMONIC.match(raw)
        if m:
            ops[m.group(1)] += 1
            continue
        for key, rx in _RES.items():
            if key in res:
                continue
            rm = rx.search(raw)
            if rm:
                res[key] = int(rm.group(1))
    return ops, res


def fmt_table(rows: list[tuple], headers: tuple) -> str:
    widths = [
        max(len(str(r[i])) for r in rows + [headers])
        for i in range(len(headers))
    ]
    out = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append(
            "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path, help="one or two .s files")
    p.add_argument("--top", type=int, default=45, help="rows in opcode table")
    args = p.parse_args()
    if len(args.files) > 2:
        p.error("at most two .s files")

    parsed = [parse(f) for f in args.files]
    names = [f.name[:28] for f in args.files]

    # ---- resource footprint ----
    print("== resources ==")
    res_keys = [
        "vgpr_count", "vgpr_spill", "sgpr_count", "sgpr_spill",
        "lds_bytes", "scratch_bytes",
    ]
    rows = [
        tuple([k] + [res.get(k, "-") for _, res in parsed]) for k in res_keys
    ]
    print(fmt_table(rows, tuple(["metric"] + names)))

    # ---- per-class summary ----
    print("\n== instruction classes ==")
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
    print(fmt_table(rows, tuple(["opcode"] + names)))
    dropped = len(ranked) - args.top
    if dropped > 0:
        print(f"... {dropped} more opcodes (raise --top to see)")


if __name__ == "__main__":
    main()
