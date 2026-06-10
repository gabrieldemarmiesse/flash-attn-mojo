#!/usr/bin/env python3
"""ptxas uniform-register allocation probe.

Generates minimal sm_90a PTX kernels — a hot loop issuing wgmma
m64n80k16 with the smem descriptor chain built in different dataflow
shapes — compiles each with ptxas, and reports how the descriptor
was allocated:

  * UR-native: built with U-ALU ops (UIADD3/ULOP3/UMOV), consumed as
    gdesc[URn] with no R2UR.
  * R2UR: built in regular registers, shuttled before each HGMMA.
  * spills: STL/LDL count + ptxas -v spill bytes (under .maxnreg
    pressure).

Purpose: bisect which PTX construct defeats ptxas's warp-uniformity
analysis — the root cause of the bwd_fa4 kernel's ~55-spill /
77-R2UR consumer loop (mojo/LLVM emits 64-bit descriptor chains;
cute's PTX keeps them 32-bit). See HANDOFF.md "the codegen wall".

Usage: python scripts/ptxas_ur_probe.py [--keep] [variant ...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PTXAS = Path(
    "/root/flash-attn-mojo/.venv/lib/python3.13/site-packages/torch/bin/ptxas"
)
NVDISASM = Path(
    "/root/.cache/uv/archive-v0/aICYSkcTP4hf2ymk/triton/backends/nvidia/"
    "bin/nvdisasm"
)

# Matrix-descriptor template for a 128B-swizzled operand:
# LBO=1, SBO=0x40, swizzle bits — taken verbatim from the kernel dump
# (or.b64 %rd, %rd, 4611686293305360386 = 0x4000004000010002 with the
# low addr bits folded; we keep template and addr separate).
DESC_TEMPLATE = 0x4000_0040_0001_0000


def wgmma(d_regs: str, a: str, b: str) -> str:
    return (
        "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
        f"wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 "
        f"{{{d_regs}}}, {a}, {b}, p, 1,  1, 0,  0;\n}}"
    )


def accum_regs(n: int, base: str = "%facc") -> str:
    return ", ".join(f"{base}{i}" for i in range(n))


def kernel(
    pre: str,
    body: str,
    extra: str,
    *,
    pressure: int = 0,
    maxnreg: int = 240,
    branch: bool = False,
    smr: bool = False,
) -> str:
    """Wrap a loop body (which must define %rdA/%rdB descriptors and
    advance them) in the toy kernel scaffold.

    pressure: number of extra live f32 chains across the loop.
    """
    ntid = 384 if branch else 128
    guard = ""
    if branch:
        # warp-specialization analog: producer wg exits (after a reg
        # dealloc like the real kernel), consumer wgs run the loop.
        guard = (
            "mov.u32 %r50, %tid.x;\n"
            "    setp.ge.u32 %p7, %r50, 128;\n"
            "    @%p7 bra $L_consumer;\n"
            "    setmaxnreg.dec.sync.aligned.u32 24;\n"
            "    bra $L_exit;\n"
            "$L_consumer:"
        )
    smr_inc = "setmaxnreg.inc.sync.aligned.u32 240;" if smr else ""
    acc = accum_regs(40)
    press_init = "\n".join(
        f"mov.f32 %fp{i}, 0f3F80000{i % 10};" for i in range(pressure)
    )
    press_iter = "\n".join(
        f"fma.rn.f32 %fp{i}, %fp{i}, %f1, %f2;" for i in range(pressure)
    )
    press_sink = "\n".join(
        f"st.global.f32 [%rd90+{4 * (40 + i)}], %fp{i};"
        for i in range(pressure)
    )
    _ = acc
    return f"""
.version 8.5
.target sm_90a
.address_size 64

.visible .entry toy(
    .param .u64 pout,
    .param .u32 niter
)
.reqntid {ntid}, 1, 1
.maxnreg {maxnreg}
{{
    .shared .align 1024 .b8 smem[131072];
    .shared .align 8 .b64 mbar;
    .reg .pred %p<8>;
    .reg .b16 %rs<4>;
    .reg .f32 %f<8>;
    .reg .f32 %facc<48>;
    .reg .f32 %fp<{max(pressure, 1) + 1}>;
    .reg .b32 %r<96>;
    .reg .b64 %rd<96>;

    ld.param.u64 %rd1, [pout];
    cvta.to.global.u64 %rd90, %rd1;
    ld.param.u32 %r1, [niter];
    mov.f32 %f1, 0f3F800000;
    mov.f32 %f2, 0f3F000000;
    {press_init}
    mov.u32 %r3, 0;            // mbarrier phase
    {guard}
    {smr_inc}
{pre}
    mov.u32 %r2, 0;            // loop counter

$L_loop:
    wgmma.fence.sync.aligned;
{body}
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;
    {press_iter}
    {extra}
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p1, %r2, %r1;
    @%p1 bra $L_loop;

    // sink the accumulators so nothing is dead
    st.global.f32 [%rd90], %facc0;
    st.global.f32 [%rd90+4], %facc39;
    {press_sink}
$L_exit:
    ret;
}}
"""


# --------------------------------------------------------------------
# Composable variant generator. A variant name is a +-joined set of
# trait flags; each trait perturbs the descriptor dataflow the way the
# real mojo or cute kernels do.
#
#   base traits (how the smem base address is derived):
#     inv    smem symbol only (loop- and thread-invariant)   [default]
#     tid7   += (tid>>7)*16384   — warpgroup index, warp-uniform
#     tid5   += (tid>>5)*16384   — warp index, warp-uniform
#     tid0   += tid*16           — DIVERGENT (negative control)
#   step traits (how per-wgmma descriptors are formed):
#     or64   or.b64 invariant_base, imm64 per step            [default]
#     b32    32-bit addr math per step, cvt+or to 64
#     carried  descriptor advanced by add.s64 across the loop (phi)
#     selp   base chosen per iteration via selp.b64 of two bases
#     phi2   base toggled between two values via a 2-target phi
#   context traits:
#     branch    loop nested under `if (tid >= 128)`
#     smr       setmaxnreg.inc 240 in the loop region
#     mix       base value also feeds a divergent st.shared
#     spin      an mbarrier-style predicated spin loop per iteration
# --------------------------------------------------------------------

ACC = accum_regs(40)


def gen_body(traits: set[str]) -> tuple[str, str, str]:
    """Returns (prologue_outside_loop, loop_body, extra_loop_carried)."""
    pre = [
        "mov.u32 %r10, smem;",
    ]
    # ---- base derivation
    if "tid7" in traits or "tid5" in traits or "tid0" in traits:
        pre.append("mov.u32 %r11, %tid.x;")
        if "tid7" in traits:
            pre.append("shr.u32 %r12, %r11, 7;")
            pre.append("shl.b32 %r13, %r12, 14;")  # *16384
        elif "tid5" in traits:
            pre.append("shr.u32 %r12, %r11, 5;")
            pre.append("shl.b32 %r13, %r12, 14;")
        else:  # tid0 — divergent
            pre.append("shl.b32 %r13, %r11, 4;")
        pre.append("add.s32 %r10, %r10, %r13;")
    if "mix" in traits:
        # divergent use of the SAME base value
        pre.append("mov.u32 %r14, %tid.x;")
        pre.append("shl.b32 %r15, %r14, 2;")
        pre.append("add.s32 %r16, %r10, %r15;")
    pre.append("cvt.u64.u32 %rd10, %r10;")
    pre.append("shr.u64 %rd11, %rd10, 4;")
    pre.append("shr.u32 %r17, %r10, 4;")  # 32-bit (addr>>4)

    body = []
    if "mix" in traits:
        body.append("st.shared.f32 [%r16], %f1;")
    if "spin" in traits:
        # the real consumer-loop construct: mbarrier.try_wait spin
        body.append("$L_spin:")
        body.append(
            "mbarrier.try_wait.parity.shared.b64 %p2, [mbar], %r3;"
        )
        body.append("@!%p2 bra $L_spin;")
    if "vspin" in traits:
        body.append("$L_vspin:")
        body.append("ld.volatile.shared.u32 %r18, [%r10];")
        body.append("setp.eq.b32 %p4, %r18, 7;")
        body.append("@%p4 bra $L_vspin;")

    extra = ""
    many = next(
        (int(t[4:]) for t in traits if re.fullmatch(r"many\d+", t)), 0
    )
    m32 = next(
        (int(t[3:]) for t in traits if re.fullmatch(r"m32\d+", t)), 0
    )
    if "tid7c" in traits:
        # the FIX shape: 32-bit extract of the warpgroup index, then
        # widen the (already-uniform) result for 64-bit chains.
        body.append("mov.u32 %r45, %tid.x;")
        body.append("shr.u32 %r46, %r45, 7;")
        body.append("shl.b32 %r47, %r46, 9;")
        body.append("and.b32 %r48, %r47, 15872;")
        body.append("cvt.u64.u32 %rd48, %r48;")
        body.append("cvt.u64.u32 %rd40, %r17;")
        body.append("add.s64 %rd41, %rd40, %rd48;")
        for k in range(8):
            body.append(f"or.b64 %rd2{k}, %rd41, {DESC_TEMPLATE + 2 * k};")
        for k in range(8):
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 8}"))
    elif "tid7w" in traits:
        # the mojo/LLVM shape: tid widened to 64-bit FIRST, then the
        # warpgroup index extracted with shr.u64 — vs tid7's 32-bit
        # shr.u32. Tests whether ptxas's tid-uniformity rule only
        # matches the 32-bit pattern.
        body.append("mov.u32 %r45, %tid.x;")
        body.append("cvt.u64.u32 %rd45, %r45;")
        body.append("shr.u64 %rd46, %rd45, 7;")
        body.append("shl.b64 %rd47, %rd46, 9;")
        body.append("and.b64 %rd48, %rd47, 15872;")
        body.append("cvt.u64.u32 %rd40, %r17;")
        body.append("add.s64 %rd41, %rd40, %rd48;")
        for k in range(8):
            body.append(f"or.b64 %rd2{k}, %rd41, {DESC_TEMPLATE + 2 * k};")
        for k in range(8):
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 8}"))
    elif "bferoot" in traits:
        # same as shflroot but the address-field extract uses
        # bfe.u32 (LLVM's canonicalization of shr+and) — suspected
        # to have no uniform-datapath counterpart.
        body.append("shfl.sync.idx.b32 %r40, %r10, 0, 31, -1;")
        body.append("bfe.u32 %r42, %r40, 4, 14;")
        body.append("cvt.u64.u32 %rd40, %r42;")
        for k in range(8):
            body.append(f"or.b64 %rd2{k}, %rd40, {DESC_TEMPLATE + 2 * k};")
        for k in range(8):
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 8}"))
    elif "shflroot" in traits:
        # In-loop shfl.idx lane-0 broadcast as the descriptor root:
        # convergent (LLVM cannot hoist), and ptxas recognizes the
        # broadcast idiom (expect ~1 R2UR per root, U-side variants).
        body.append("shfl.sync.idx.b32 %r40, %r10, 0, 31, -1;")
        body.append("shr.u32 %r41, %r40, 4;")
        body.append("and.b32 %r42, %r41, 16376;")
        body.append("cvt.u64.u32 %rd40, %r42;")
        for k in range(8):
            body.append(f"or.b64 %rd2{k}, %rd40, {DESC_TEMPLATE + 2 * k};")
        for k in range(8):
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 8}"))
    elif m32:
        # N distinct 32-bit address roots live across the loop; the
        # 64-bit descriptor is materialized transiently per use
        # (cute's discipline: 1 UR per root instead of 2).
        for d in range(m32):
            pre.append(f"add.s32 %r{20 + d}, %r17, {2 * d};")
        for d in range(m32):
            body.append(f"cvt.u64.u32 %rd20, %r{20 + d};")
            body.append(f"or.b64 %rd20, %rd20, {DESC_TEMPLATE};")
            body.append(
                f"cvt.u64.u32 %rd21, %r{20 + (d + 1) % m32};"
            )
            body.append(f"or.b64 %rd21, %rd21, {DESC_TEMPLATE};")
            body.append(wgmma(ACC, "%rd20", "%rd21"))
    elif many:
        # N distinct descriptors live across the loop: UR-file
        # pressure probe (63 URs/warp; each 64-bit desc = 2).
        for d in range(many):
            pre.append(
                f"or.b64 %rd{20 + d}, %rd11, {DESC_TEMPLATE + 2 * d};"
            )
        for d in range(many):
            body.append(
                wgmma(ACC, f"%rd{20 + d}", f"%rd{20 + (d + 1) % many}")
            )
    elif "carried" in traits:
        pre.append(f"or.b64 %rd30, %rd11, {DESC_TEMPLATE};")
        for k in range(4):
            body.append(f"add.s64 %rd2{k}, %rd30, {2 * k};")
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 4}"))
        # advance the carried descriptor each iteration (and wrap so
        # it stays loop-variant without growing unbounded)
        extra = (
            "add.s64 %rd30, %rd30, 64;\n"
            "and.b64 %rd31, %rd30, 1023;\n"
            f"or.b64 %rd30, %rd31, {DESC_TEMPLATE};"
        )
    elif "selp" in traits or "phi2" in traits:
        pre.append(f"or.b64 %rd30, %rd11, {DESC_TEMPLATE};")
        pre.append(f"or.b64 %rd31, %rd11, {DESC_TEMPLATE + 1280};")
        if "selp" in traits:
            body.append("and.b32 %r19, %r2, 1;")
            body.append("setp.eq.b32 %p3, %r19, 0;")
            body.append("selp.b64 %rd32, %rd30, %rd31, %p3;")
        else:  # phi2: toggle via swap each iteration
            pre.append("mov.b64 %rd32, %rd30;")
            pre.append("mov.b64 %rd33, %rd31;")
            extra = (
                "mov.b64 %rd34, %rd32;\n"
                "mov.b64 %rd32, %rd33;\n"
                "mov.b64 %rd33, %rd34;"
            )
        for k in range(4):
            body.append(f"add.s64 %rd2{k}, %rd32, {2 * k};")
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 4}"))
    elif "b32" in traits:
        for k in range(4):
            body.append(f"add.s32 %r2{k}, %r17, {2 * k};")
            body.append(f"cvt.u64.u32 %rd2{k}, %r2{k};")
            body.append(f"or.b64 %rd2{k}, %rd2{k}, {DESC_TEMPLATE};")
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 4}"))
    else:  # or64 default
        for k in range(4):
            body.append(f"or.b64 %rd2{k}, %rd11, {DESC_TEMPLATE + 2 * k};")
            body.append(wgmma(ACC, f"%rd2{k}", f"%rd2{(k + 1) % 4}"))

    return "\n".join(pre), "\n".join(body), extra


def analyze(name: str, ptx: str, keep: bool) -> dict:
    with tempfile.TemporaryDirectory() as td:
        ptx_path = Path(td) / f"{name}.ptx"
        cubin = Path(td) / f"{name}.cubin"
        ptx_path.write_text(ptx)
        if keep:
            Path(f"/tmp/urprobe_{name}.ptx").write_text(ptx)
        r = subprocess.run(
            [str(PTXAS), "-arch=sm_90a", "-v", str(ptx_path), "-o", str(cubin)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return {"name": name, "error": r.stderr.strip()[:400]}
        info = r.stderr
        spill = re.search(r"(\d+) bytes spill stores", info)
        regs = re.search(r"Used (\d+) registers", info)
        sass = subprocess.run(
            [str(NVDISASM), "-c", str(cubin)], capture_output=True, text=True
        ).stdout
        if keep:
            Path(f"/tmp/urprobe_{name}.sass").write_text(sass)
        return {
            "name": name,
            "regs": int(regs.group(1)) if regs else -1,
            "spill_bytes": int(spill.group(1)) if spill else 0,
            "R2UR": len(re.findall(r"\bR2UR\b", sass)),
            "U_alu": len(re.findall(r"\bU(?:IADD3|LOP3|MOV|SHF|LEA)\b", sass)),
            "STL_LDL": len(re.findall(r"\b(?:STL|LDL)\b", sass)),
            "gdesc_UR": len(re.findall(r"gdesc\[UR\d+\]", sass)),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=[])
    ap.add_argument("--keep", action="store_true", help="dump ptx/sass to /tmp")
    ap.add_argument("--pressure", type=int, default=0)
    args = ap.parse_args()

    default_sweep = [
        "inv",
        "tid0",
        "spin",
        "vspin",
        "branch",
        "branch+smr",
        "tid7+branch+smr",
        "spin+branch+smr",
        "tid7+spin+branch+smr",
        "tid7+selp+spin+branch+smr+mix",
        "tid7+carried+spin+branch+smr+mix",
    ]
    names = args.variants or default_sweep
    rows = []
    for n in names:
        traits = set(n.split("+"))
        pre, body, extra = gen_body(traits)
        ptx = kernel(
            pre,
            body,
            extra,
            pressure=args.pressure,
            branch="branch" in traits,
            smr="smr" in traits,
        )
        rows.append(analyze(n.replace("+", "_"), ptx, args.keep))

    hdr = ["name", "regs", "spill_bytes", "R2UR", "U_alu", "STL_LDL", "gdesc_UR"]
    print("  ".join(f"{h:>12s}" for h in hdr))
    for row in rows:
        if "error" in row:
            print(f"{row['name']:>12s}  ERROR: {row['error']}")
            continue
        print("  ".join(f"{str(row[h]):>12s}" for h in hdr))


if __name__ == "__main__":
    main()
