#!/usr/bin/env bash
# Wrap ncu around scripts/profile_bench.py.
#
# Handles the friction points discovered during the bwd perf work:
#   * Pulls ncu via `pixi exec --spec nsight-compute=2024.3.2` (no
#     system install needed).
#   * Detects /proc/driver/nvidia/params:RmProfilingAdminOnly and
#     re-execs under sudo if the gate is locked; gives a clear error
#     if sudo is missing.
#   * Resolves pixi and uv to absolute paths so they survive sudo's
#     PATH wipe.
#   * Wraps capture iters in cudaProfilerStart/Stop (via the bench
#     script's --profile flag) and passes --profile-from-start no to
#     ncu, so JIT compile and warmup launches are excluded
#     automatically — no manual --launch-skip arithmetic.
#
# Usage:
#   scripts/profile_kernel.sh [profiler opts] -- [bench opts]
#
# Profiler opts (all optional, with sensible defaults):
#   --kernel KIND       fwd | bwd | bwd-main | bwd-preprocess | bwd-convert
#                       (default: matches --kind from bench opts)
#   --filter REGEX      ncu kernel-name regex (overrides --kernel)
#   --set NAME          ncu section set: full | basic | detailed | source
#                       (default: full)
#   --output PATH       output report path without extension
#                       (default: /tmp/<filter>_prof)
#   --iters N           captured iterations (default: 1)
#   --warmup N          warmup iterations before profiling (default: 3)
#   --no-sudo           never use sudo (will fail if gate is locked)
#
# Bench opts (forwarded to scripts/profile_bench.py after `--`):
#   --kind fwd|bwd                       (required)
#   --shape B,L,H,D                      (default: 1,1024,8,64)
#   --causal, --dtype bf16|fp16          (optional)
#
# Examples:
#   scripts/profile_kernel.sh --kernel bwd-main -- --kind bwd --shape 1,1024,8,64
#   scripts/profile_kernel.sh --set basic -- --kind fwd --shape 1,2048,8,64 --causal
#   scripts/profile_kernel.sh --filter 'flash_fwd' -- --kind fwd  # upstream

set -euo pipefail

# ---- parse profiler opts ----
KERNEL=""
FILTER=""
SET_NAME="full"
OUTPUT=""
ITERS=1
WARMUP=3
USE_SUDO="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL="$2"; shift 2 ;;
        --filter) FILTER="$2"; shift 2 ;;
        --set) SET_NAME="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --no-sudo) USE_SUDO="no"; shift ;;
        --) shift; break ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown profiler option '$1' (forward bench opts after \`--\`)" >&2
            exit 2
            ;;
    esac
done

BENCH_ARGS=("$@")

# ---- derive filter from --kernel if --filter not set ----
if [[ -z "$FILTER" && -n "$KERNEL" ]]; then
    case "$KERNEL" in
        fwd)            FILTER='fwd_fa4_kernel' ;;
        bwd)            FILTER='bwd_' ;;
        bwd-main)       FILTER='bwd_main_kernel' ;;
        bwd-preprocess) FILTER='bwd_preprocess_kernel' ;;
        bwd-convert)    FILTER='bwd_convert_kernel' ;;
        *)
            echo "error: unknown --kernel '$KERNEL' (use fwd|bwd|bwd-main|bwd-preprocess|bwd-convert)" >&2
            exit 2
            ;;
    esac
fi

# ---- fall back to inferring filter from --kind in BENCH_ARGS ----
if [[ -z "$FILTER" ]]; then
    for ((i=0; i<${#BENCH_ARGS[@]}; i++)); do
        if [[ "${BENCH_ARGS[$i]}" == "--kind" ]] && (( i+1 < ${#BENCH_ARGS[@]} )); then
            case "${BENCH_ARGS[$i+1]}" in
                fwd) FILTER='fwd_fa4_kernel' ;;
                bwd) FILTER='bwd_' ;;
            esac
            break
        fi
    done
fi

if [[ -z "$FILTER" ]]; then
    echo "error: must pass --kernel, --filter, or include --kind in bench args" >&2
    exit 2
fi

# ---- default output path ----
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="/tmp/${FILTER//[^A-Za-z0-9]/_}_prof"
fi

# ---- resolve absolute paths (sudo wipes PATH) ----
PIXI="$(command -v pixi || true)"
UV="$(command -v uv || true)"
if [[ -z "$PIXI" ]]; then
    echo "error: pixi not found on PATH; install from https://pixi.sh" >&2
    exit 1
fi
if [[ -z "$UV" ]]; then
    echo "error: uv not found on PATH; install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

# ---- check the profiling permission gate ----
GATE=""
if [[ -r /proc/driver/nvidia/params ]]; then
    GATE="$(awk '/^RmProfilingAdminOnly:/ {print $2}' /proc/driver/nvidia/params)"
fi

SUDO_PREFIX=()
if [[ "$GATE" == "1" ]]; then
    case "$USE_SUDO" in
        no)
            echo "error: RmProfilingAdminOnly=1 but --no-sudo passed; ncu will return ERR_NVGPUCTRPERM" >&2
            echo "       unlock with: rmmod nvidia_* && modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0" >&2
            exit 1
            ;;
        auto)
            if sudo -n true 2>/dev/null; then
                SUDO_PREFIX=(sudo -E)
            else
                echo "error: RmProfilingAdminOnly=1 and passwordless sudo unavailable." >&2
                echo "       Either configure sudoers, run under sudo manually, or pass --no-sudo" >&2
                echo "       after unlocking the driver gate." >&2
                exit 1
            fi
            ;;
    esac
fi

# ---- run ncu ----
echo "[profile_kernel] filter='$FILTER' set='$SET_NAME' output='${OUTPUT}.ncu-rep'" >&2

"${SUDO_PREFIX[@]}" "$PIXI" exec --spec nsight-compute=2024.3.2 -- ncu \
    --target-processes all \
    --kernel-name "regex:${FILTER}" \
    --profile-from-start no \
    --set "$SET_NAME" \
    --force-overwrite \
    -o "$OUTPUT" \
    "$UV" run --extra nvidia python "$(dirname "$0")/profile_bench.py" \
        --profile \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        "${BENCH_ARGS[@]}"

echo "[profile_kernel] report: ${OUTPUT}.ncu-rep" >&2
echo "[profile_kernel] open:   ncu-ui ${OUTPUT}.ncu-rep" >&2
echo "[profile_kernel] summary: scripts/profile_summary.sh ${OUTPUT}.ncu-rep" >&2
