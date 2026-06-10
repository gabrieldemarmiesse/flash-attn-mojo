#!/usr/bin/env bash
# Master iteration script for the FA4-vs-Mojo forward race.
#
# One invocation =
#   1. clear the flash_attn_mojo JIT cache (the mojo compiler's own
#      cache is untouched — it is trusted to be incremental) and
#      recompile the fwd_fa4 kernel from source;
#   2. benchmark mojo vs Tri Dao FA4 on one big shape (kernel-only
#      GPU time via CUPTI) + correctness check;
#   3. dump the mojo kernel's PTX to ptx/mojo_fwd_fa4.ptx (wired via
#      the MOJO_DUMP_PTX env var -> -D define -> compile_function);
#   4. print instruction-mix stats of FA4's PTX vs the mojo PTX;
#   5. (unless --no-ncu) capture + print the main ncu stats for both
#      kernels side by side.
#
# Usage:
#   scripts/master_bench.sh [--kind fwd|bwd] [--shape B,S,H,D]
#                           [--iters N] [--no-ncu]
#                           [--ncu-set basic|detailed|full]
#                           [--refresh-fa4-ptx] [--no-check]
#
# Typical loop: edit src/flash_attn_mojo/{fwd,bwd}_fa4/kernel.mojo,
# run scripts/master_bench.sh [--kind bwd], read the summary, repeat.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KIND="fwd"
SHAPE="2,8192,16,128"
ITERS=20
RUN_NCU=1
NCU_SET="basic"
REFRESH_FA4_PTX=0
CHECK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind) KIND="$2"; shift 2 ;;
        --shape) SHAPE="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --no-ncu) RUN_NCU=0; shift ;;
        --ncu-set) NCU_SET="$2"; shift 2 ;;
        --refresh-fa4-ptx) REFRESH_FA4_PTX=1; shift ;;
        --no-check) CHECK=0; shift ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$KIND" == "fwd" ]]; then
    FA4_PTX="$ROOT/reference_ptx/fa4_fwd_sm90_bf16_hdim128_noncausal.ptx"
    MOJO_PTX="$ROOT/ptx/mojo_fwd_fa4.ptx"
    FA4_NCU_FILTER='FlashAttentionForwardSm90'
    MOJO_NCU_FILTER='fwd_fa4_kernel'
else
    FA4_PTX="$ROOT/reference_ptx/fa4_bwd_sm90_bf16_hdim128_noncausal.ptx"
    MOJO_PTX="$ROOT/ptx/mojo_bwd_fa4.ptx"
    # ncu compares the *main* bwd kernel (>95% of bwd time).
    FA4_NCU_FILTER='FlashAttentionBackwardSm90'
    MOJO_NCU_FILTER='bwd_main_kernel'
fi
mkdir -p "$ROOT/ptx"

UV="$(command -v uv)"
CHECK_FLAG=(); [[ "$CHECK" == 1 ]] && CHECK_FLAG=(--check)

step() { printf '\n\033[1m==== %s ====\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 1
step "1. clear flash_attn_mojo JIT cache + recompile"
rm -rf ~/.cache/flash_attn_mojo
# ------------------------------------------------------------- 2+3
# A single bench run compiles the kernel (cache was just cleared),
# dumps its PTX (MOJO_DUMP_PTX define), checks correctness vs fp32
# SDPA + FA4, and measures kernel time.
step "2+3. mojo ($KIND): compile, dump PTX, check, bench"
MOJO_RESULT="$(MOJO_DUMP_PTX="$MOJO_PTX" "$UV" run python scripts/bench_fa4.py \
    --impl mojo --kind "$KIND" --shape "$SHAPE" --iters "$ITERS" "${CHECK_FLAG[@]}" | tee /dev/stderr | grep ^RESULT)"

step "2b. fa4 ($KIND) bench"
if [[ "$REFRESH_FA4_PTX" == 1 ]]; then
    TMP_PTX_DIR="$(mktemp -d)"
    FA4_RESULT="$(CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR="$TMP_PTX_DIR" \
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="$(mktemp -d)" \
        "$UV" run python scripts/bench_fa4.py \
        --impl fa4 --kind "$KIND" --shape "$SHAPE" --iters "$ITERS" "${CHECK_FLAG[@]}" | tee /dev/stderr | grep ^RESULT)"
    tr -d '\000' < "$TMP_PTX_DIR"/cutlass*${FA4_NCU_FILTER}*.ptx > "$FA4_PTX"
    echo "[master_bench] refreshed $FA4_PTX"
else
    FA4_RESULT="$("$UV" run python scripts/bench_fa4.py \
        --impl fa4 --kind "$KIND" --shape "$SHAPE" --iters "$ITERS" "${CHECK_FLAG[@]}" | tee /dev/stderr | grep ^RESULT)"
fi

# ---------------------------------------------------------------- 4
step "4. PTX instruction mix: fa4 (reference) vs mojo"
"$UV" run python scripts/ptx_stats.py "$FA4_PTX" "$MOJO_PTX"

# ---------------------------------------------------------------- 5
if [[ "$RUN_NCU" == 1 ]]; then
    step "5. ncu ($NCU_SET set): capture both kernels"
    PIXI="$(command -v pixi || true)"
    if [[ -z "$PIXI" ]]; then
        echo "[master_bench] pixi not found — skipping ncu section" >&2
    else
        NCU_SPEC=(exec --spec nsight-compute=2024.3.2 -- ncu)
        # RmProfilingAdminOnly gate -> sudo -E (same logic as
        # profile_kernel.sh).
        GATE=""
        [[ -r /proc/driver/nvidia/params ]] && \
            GATE="$(awk '/^RmProfilingAdminOnly:/ {print $2}' /proc/driver/nvidia/params)"
        SUDO_PREFIX=()
        if [[ "$GATE" == "1" ]]; then
            if sudo -n true 2>/dev/null; then
                SUDO_PREFIX=(sudo -E)
            else
                echo "[master_bench] RmProfilingAdminOnly=1, no passwordless sudo — skipping ncu" >&2
                PIXI=""
            fi
        fi
        if [[ -n "$PIXI" ]]; then
            for IMPL in fa4 mojo; do
                case "$IMPL" in
                    fa4)  FILT="$FA4_NCU_FILTER" ;;
                    mojo) FILT="$MOJO_NCU_FILTER" ;;
                esac
                echo "[master_bench] ncu capture: $IMPL (filter $FILT)"
                "${SUDO_PREFIX[@]}" "$PIXI" "${NCU_SPEC[@]}" \
                    --target-processes all \
                    --kernel-name "regex:${FILT}" \
                    --launch-count 1 \
                    --profile-from-start no \
                    --set "$NCU_SET" \
                    --force-overwrite -o "/tmp/master_bench_${IMPL}" \
                    "$UV" run python scripts/bench_fa4.py \
                        --impl "$IMPL" --kind "$KIND" --shape "$SHAPE" --profile \
                        --iters 1 --warmup 3 > /dev/null
            done
            "$UV" run python scripts/ncu_compare.py \
                --ncu "$PIXI exec --spec nsight-compute=2024.3.2 -- ncu" \
                /tmp/master_bench_fa4.ncu-rep /tmp/master_bench_mojo.ncu-rep
        fi
    fi
fi

# ---------------------------------------------------------- summary
step "summary"
echo "$FA4_RESULT"
echo "$MOJO_RESULT"
FA4_US="$(sed -E 's/.* us=([0-9.]+).*/\1/' <<< "$FA4_RESULT")"
MOJO_US="$(sed -E 's/.* us=([0-9.]+).*/\1/' <<< "$MOJO_RESULT")"
python3 - "$FA4_US" "$MOJO_US" <<'EOF'
import sys
fa4, mojo = float(sys.argv[1]), float(sys.argv[2])
print(f"mojo/fa4 ratio: {mojo / fa4:.3f}x  ({'mojo SLOWER' if mojo > fa4 else 'mojo FASTER'})")
EOF
echo "mojo PTX: ptx/mojo_fwd_fa4.ptx   fa4 PTX: reference_ptx/$(basename "$FA4_PTX")"
