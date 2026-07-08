#!/usr/bin/env bash
# master_bench_rocm.sh — the ROCm/MI300X analog of master_bench.sh.
#
# One invocation =
#   1. recompile the mojo v0 forward kernel from source
#      (bench/bench_mojo_rocm.mojo -> bench/build/bench_mojo_rocm);
#   2. run the mojo v0 (fp32 CPU correctness check + wall-clock time)
#      and the CK reference (kernel-only device time via roctracer,
#      the CUPTI analog) on one shape, print both + ratio;
#   3. copy the mojo kernel's AMDGCN ISA dump (the PTX analog, written
#      by dump_asm on every run) into asm/;
#   4. print the GCN instruction-mix + resource footprint of the mojo
#      kernel (spills, LDS, matrix-op count — the canaries);
#   5. (unless --no-prof) re-time the mojo kernel under rocprofv3 for a
#      kernel-only number + launch resources (the ncu-stats analog).
#
# There is no FlashAttention-4 / CuTe on AMD, so the reference baseline is
# Tri Dao's `flash_attn` built with the Composable Kernel (CK) backend —
# the fastest attention kernel on this MI300X (it beat both PyTorch SDPA
# and the AMD Triton FA2 backend; see scripts/README.md for that
# comparison). CK is the ONLY reference here: if `flash_attn` is not
# installed the harness errors out (build it per scripts/README.md).
# The mojo lane is v0 — a hand-vectorized SIMD kernel that does NOT use
# the CDNA matrix cores yet, so expect it far behind the reference; this
# harness is the M0 milestone (correctness + measurement plumbing), the
# analog of the Metal race's M0. See bench/bench_mojo_rocm.mojo.
#
# Measurement protocol (MI300X edition): this is a virtualized (SR-IOV)
# MI300X with no user clock-locking; trust only interleaved runs and the
# printed spreads. The mojo lane's wall-clock time includes mojo's
# per-enqueue dispatch overhead (the reference's kernel-only time does
# not) — step 5's rocprofv3 numbers remove that asymmetry.
#
# Usage:
#   scripts/master_bench_rocm.sh [--seq N] [--head-dim 64|128]
#       [--heads N] [--batch N] [--iters N] [--quick] [--no-prof]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEQ=4096
HDIM=128
HEADS=16
BATCH=1
ITERS=20
PROF=1
QUICK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seq) SEQ="$2"; shift 2 ;;
        --head-dim) HDIM="$2"; shift 2 ;;
        --heads) HEADS="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --no-prof) PROF=0; shift ;;
        -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$QUICK" == 1 ]]; then SEQ=1024; HEADS=8; ITERS=5; fi

MOJO_BIN="$ROOT/bench/build/bench_mojo_rocm"
MOJO_ASM="/tmp/mojo_fwd_rocm_d${HDIM}.s"
UV_PY="$ROOT/.venv/bin/python"

step() { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1
step "1. build mojo v0 kernel (recompile from source)"
mkdir -p "$ROOT/bench/build" "$ROOT/asm"
"$ROOT/.venv/bin/mojo" build bench/bench_mojo_rocm.mojo -o "$MOJO_BIN"

# ------------------------------------------------------------- 2+3
step "2. mojo v0 ($SEQ x $HEADS x $HDIM): run + fp32 correctness"
MOJO_JSON="$("$MOJO_BIN" --seq "$SEQ" --head-dim "$HDIM" --heads "$HEADS" \
    --iters "$ITERS" --check | tee /dev/stderr | grep '^{')"

step "2b. CK reference (kernel-only): bench + correctness"
# CK-flash is the only reference baseline. Require it — error out (do not
# fall back) if flash_attn is not installed.
if ! "$UV_PY" -c "import flash_attn" 2>/dev/null; then
    echo "[master_bench_rocm] ERROR: flash_attn (Composable Kernel backend) is not installed." >&2
    echo "  CK is the required reference baseline — build it per scripts/README.md." >&2
    exit 1
fi
REF_RESULT="$("$UV_PY" scripts/bench_rocm.py \
    --batch "$BATCH" --seq "$SEQ" --heads "$HEADS" --head-dim "$HDIM" \
    --iters "$ITERS" --check | tee /dev/stderr | grep '^RESULT')"

# ---------------------------------------------------------------- 3
step "3. AMDGCN ISA dump -> asm/"
if [[ -f "$MOJO_ASM" ]]; then
    cp -f "$MOJO_ASM" "$ROOT/asm/mojo_fwd_rocm_d${HDIM}.s"
    echo "[master_bench_rocm] copied $MOJO_ASM -> asm/mojo_fwd_rocm_d${HDIM}.s"
else
    echo "[master_bench_rocm] WARNING: $MOJO_ASM not found" >&2
fi

# ---------------------------------------------------------------- 4
step "4. GCN instruction mix + resources (mojo v0 kernel)"
"$UV_PY" scripts/gcn_opmix.py "$ROOT/asm/mojo_fwd_rocm_d${HDIM}.s" --top 20

# ---------------------------------------------------------------- 5
# Only the mojo lane needs rocprofv3: it gives the kernel-only device
# time (removing mojo's dispatch-overhead asymmetry vs the reference)
# plus the launch resource footprint. The reference is already measured
# kernel-only by torch.profiler in step 2b, and torch crashes under
# rocprofv3 (roctracer double-instrumentation), so we don't re-profile it.
MOJO_CSV=""
if [[ "$PROF" == 1 ]]; then
    step "5. rocprofv3 kernel stats: mojo v0 (kernel-only + resources)"
    ROCPROF="$(command -v rocprofv3 || true)"
    if [[ -z "$ROCPROF" ]]; then
        echo "[master_bench_rocm] rocprofv3 not found — skipping" >&2
    else
        rm -rf /tmp/mb_rocm_mojo
        "$ROCPROF" --kernel-trace --output-format csv -d /tmp/mb_rocm_mojo -- \
            "$MOJO_BIN" --seq "$SEQ" --head-dim "$HDIM" --heads "$HEADS" \
            --iters 3 --dispatches 3 >/dev/null 2>&1 || true
        MOJO_CSV="$(find /tmp/mb_rocm_mojo -name '*kernel_trace.csv' 2>/dev/null | head -1)"
        if [[ -n "$MOJO_CSV" ]]; then
            "$UV_PY" scripts/rocprof_summary.py "$MOJO_CSV"
        else
            echo "[master_bench_rocm] rocprof CSV missing — skipping summary" >&2
        fi
    fi
fi

# ---------------------------------------------------------- summary
step "summary"
echo "$REF_RESULT"
echo "$MOJO_JSON"
REF_US="$(sed -E 's/.* us=([0-9.]+).*/\1/' <<< "$REF_RESULT")"
"$UV_PY" - "$REF_US" "$MOJO_JSON" "$MOJO_CSV" <<'EOF'
import csv, json, sys
ref_us = float(sys.argv[1])
mojo = json.loads(sys.argv[2])
mojo_csv = sys.argv[3] if len(sys.argv) > 3 else ""
mojo_wall = mojo["min_us"]
err = mojo.get("check_max_error")

# mojo kernel-only mean from the rocprof trace (drop first as warmup).
mojo_kern = None
if mojo_csv:
    try:
        rows = list(csv.DictReader(open(mojo_csv)))
        durs = [
            (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
            for r in rows
        ]
        if len(durs) > 1:
            durs = durs[1:]
        if durs:
            mojo_kern = sum(durs) / len(durs)
    except Exception:
        pass

print(f"reference CK (kernel-only): {ref_us:8.1f} us")
if err is not None:
    print(f"mojo v0   (wall-clock):    {mojo_wall:8.1f} us   "
          f"(fp32 max_err={err:.2e})")
else:
    print(f"mojo v0   (wall-clock):    {mojo_wall:8.1f} us")
if mojo_kern is not None:
    print(f"mojo v0   (kernel-only):   {mojo_kern:8.1f} us")
    r = mojo_kern / ref_us
    print(f"mojo/ref ratio (kernel-only): {r:.2f}x  "
          f"({'mojo SLOWER' if r > 1 else 'mojo FASTER'})")
else:
    r = mojo_wall / ref_us
    print(f"mojo/ref ratio (wall vs kernel): {r:.2f}x  "
          f"({'mojo SLOWER' if r > 1 else 'mojo FASTER'})")
EOF
echo "mojo ISA: asm/mojo_fwd_rocm_d${HDIM}.s"
