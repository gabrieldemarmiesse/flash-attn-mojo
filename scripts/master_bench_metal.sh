#!/usr/bin/env bash
# master_bench_metal.sh — the Metal-race analog of master_bench.sh.
#
# Builds all three forward-attention lanes (mojo v0, MFA, ccv-MFA),
# gates on correctness vs each lane's fp32/fp64 CPU reference, runs the
# process-interleaved bench matrix, refreshes the mojo kernel's AIR dump
# in air/ and prints the op-mix diff vs the committed reference AIR.
#
# Usage:
#   scripts/master_bench_metal.sh [--quick] [--full] [--impls mojo,mfa,ccv]
#                                 [--no-diff] [--profile IMPL]
#
#   --quick        one shape (S=1024, D=128), 2 rounds
#   --full         adds S=8192 to the default S=4096 matrix
#   --profile IMPL one extra short run of IMPL under
#                  `xctrace record --template 'Metal System Trace'`,
#                  then prints per-encoder GPU intervals
#
# Measurement protocol (Apple-GPU edition — see METAL_PLAN.md):
# no clock locking exists on macOS; trust only interleaved runs,
# >=5 dispatches per command buffer, warmed-up steady state, and the
# printed per-round spreads. The mojo lane times wall-clock around
# enqueue+sync (its only reliable bracket today) and so carries its
# dispatch overhead; the references report command-buffer GPU time.

set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
FULL=0
IMPLS="mojo,mfa,ccv"
DIFF=1
PROFILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --full) FULL=1 ;;
    --impls) IMPLS="$2"; shift ;;
    --no-diff) DIFF=0 ;;
    --profile) PROFILE="$2"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

PY="uv run python"

echo "=== build ==="
(cd reference_air/mfa/bench_mfa && swift build -c release)
(cd reference_air/ccv/bench_ccv && bash build.sh)
mkdir -p bench/build
.venv/bin/mojo build bench/bench_mojo_metal.mojo -o bench/build/bench_mojo_metal

echo
echo "=== correctness gate (S=1024, H=16, fp16, strided-row CPU refs) ==="
for D in 64 128; do
  $PY scripts/bench_metal.py --seq 1024 --heads 16 --head-dim "$D" \
    --impls "$IMPLS" --rounds 1 --iters 2 --check
done

echo
echo "=== bench matrix (interleaved, 3 rounds x 5 iters, 5 dispatches/cb) ==="
mkdir -p air
if [[ $QUICK -eq 1 ]]; then
  SHAPES=("1024 128")
else
  SHAPES=("4096 64" "4096 128")
  [[ $FULL -eq 1 ]] && SHAPES+=("8192 64" "8192 128")
fi
ROUNDS=$([[ $QUICK -eq 1 ]] && echo 2 || echo 3)
for shape in "${SHAPES[@]}"; do
  read -r S D <<<"$shape"
  $PY scripts/bench_metal.py --seq "$S" --heads 16 --head-dim "$D" \
    --impls "$IMPLS" --rounds "$ROUNDS" --iters 5 \
    --jsonl air/bench_metal_results.jsonl
done

# The mojo CLI rewrites its AIR (textual LLVM IR) on every run.
cp -f /tmp/mojo_fwd_metal_d64.air.ll air/ 2>/dev/null || true
cp -f /tmp/mojo_fwd_metal_d128.air.ll air/ 2>/dev/null || true

if [[ $DIFF -eq 1 ]]; then
  echo
  echo "=== AIR op-mix: mojo vs references (static IR, whole kernel) ==="
  for D in 64 128; do
    if [[ -f "air/mojo_fwd_metal_d$D.air.ll" ]]; then
      for ref in mfa ccv; do
        echo "--- d$D: $ref -> mojo (top 12 by |delta|) ---"
        $PY scripts/air_opmix.py "reference_air/$ref/fwd_d$D.air" \
          "air/mojo_fwd_metal_d$D.air.ll" --top 12 || true
      done
    fi
  done
fi

if [[ -n "$PROFILE" ]]; then
  echo
  echo "=== xctrace profile: $PROFILE (S=4096 D=128 H=16, 3 iters) ==="
  TRACE=$(mktemp -d)/bench.trace
  case "$PROFILE" in
    mojo) BIN=(bench/build/bench_mojo_metal --seq 4096 --head-dim 128 --heads 16 --iters 3 --warmup 1) ;;
    mfa)  BIN=(reference_air/mfa/bench_mfa/.build/release/bench_mfa --seq 4096 --head-dim 128 --heads 16 --iters 3 --warmup 1) ;;
    ccv)  BIN=(reference_air/ccv/bench_ccv/bench_ccv_attn --r 4096 --c 4096 --hq 16 --hk 16 --d 128 --iterations 3 --warmup 1) ;;
    *) echo "unknown --profile impl: $PROFILE" >&2; exit 2 ;;
  esac
  xctrace record --template "Metal System Trace" --output "$TRACE" \
    --launch -- "${BIN[@]}"
  $PY scripts/xctrace_gpu_intervals.py "$TRACE" || true
  echo "trace kept at: $TRACE (open in Instruments for the GUI view)"
fi

echo
echo "done. raw results: air/bench_metal_results.jsonl"
