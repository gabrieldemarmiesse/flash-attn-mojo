# reference_air/ — the Metal-race analog of reference_ptx/

Committed AIR (Apple IR — LLVM bitcode, the deepest public artifact;
there is no SASS analog on Apple GPUs) + generated MSL sources for the
two reference forward-attention implementations, plus the bench CLI
wrappers `scripts/master_bench_metal.sh` shells out to.

Everything below was generated/measured on the canonical machine:
Apple M4 Mac mini (10-core GPU, 16 GB), macOS 26.3.2, Xcode 26 with
the Metal Toolchain component installed
(`xcodebuild -downloadComponent MetalToolchain`).

## Layout

- `mfa/fwd_d{64,128}.metal|.air` — philipturner/metal-flash-attention
  (Swift, runtime MSL source-generator), forward, fp16 inputs,
  non-causal, single-head. Extracted with
  `bench_mfa --dump-source`, compiled `xcrun metal -c` (plain
  defaults = MFA's runtime `options: nil`: fast math ON).
- `mfa/bench_mfa/` — SwiftPM CLI wrapper (`swift build -c release`).
  MHA is emulated per MFA's docs: one dispatch per head, per-head
  buffers (heads concurrent, repeats hazard-serialized).
- `mfa/bench_results.jsonl` — the original single-head bench matrix.
- `ccv/fwd_d{64,128}.metal|.air` — ccv's C++ MFA port
  (`ccv/lib/nnc/mfa`, liuliu's extension of MFA: causal/GQA/batch/
  varlen/mask/sliding-window/sinks). Extracted with
  `bench_ccv_attn --dump-source`, compiled
  `xcrun metal -std=metal3.1`.
- `ccv/bench_ccv/` — standalone C++ driver + `build.sh`
  (self-sufficient against a fresh ccv clone: applies the `\01air`
  patch and compiles the 8 needed objects from source).

## The macOS 26 `__asm("air.*")` patch

Metal 4's compiler rejects `__asm` labels containing dots, which
breaks the `air.simdgroup_async_copy_*` / `air.wait_simdgroup_events`
declarations both MFA-family generators emit. Fix (verified
end-to-end, byte-identical AIR symbols): prefix the label with LLVM's
literal-symbol escape — `__asm("\01air.simdgroup_async_copy_2d...")`.

- MFA clone: 5-line working-tree patch in
  `Sources/FlashAttention/GEMM/GEMMHeaders.swift`; `bench_mfa`
  re-applies the substitution at runtime, so it works even against a
  pristine clone.
- ccv clone: same 5 sites in `lib/nnc/mfa/kernels/GEMMHeaders.cpp`;
  `bench_ccv/build.sh` re-applies it idempotently.

Without the patch, ccv silently falls back to a precompiled
macOS-13-era metallib (`libmfamacos13-v1.0.2-b.metallib`) — different
kernel, wrong target numbers. `bench_ccv_attn` prints
`runtime_compile_ok=1` on stderr when the live generator is in use.

## Target numbers (what the mojo kernel must beat)

Forward, non-causal, fp16 inputs (O is fp32 — both references page O
as fp32 by design), B=1 H=16, min over 3x5 interleaved trials,
kernel-only command-buffer GPU time, 5 dispatches/command-buffer
(single-dispatch command buffers flap 5-50x on M4 — power states):

| S    | D   | MFA (us) | ccv (us) | ccv GFLOPS |
|------|-----|----------|----------|------------|
| 1024 | 64  | 1651     | 1161     | 3699       |
| 1024 | 128 | 3270     | 2354     | 3648       |
| 4096 | 64  | 19618    | 18241    | 3766       |
| 4096 | 128 | 40223    | 36882    | 3726       |
| 8192 | 64  | 75934    | 72967    | 3767       |
| 8192 | 128 | 156242   | 149259   | 3683       |

GFLOPS = 4·S²·D·H / t. ccv beats upstream MFA by 4-8% everywhere.
Both kernels' AIR shows the same engine: 8x8 simdgroup-matrix MMAs
(`air.simdgroup_matrix_8x8_multiply_accumulate`, fp16 accumulate for
S=QK^T, fp32 for O) + `simdgroup_async_copy` threadgroup staging.

## Regeneration

```bash
cd reference_air/mfa/bench_mfa && swift build -c release
./.build/release/bench_mfa --seq 1024 --head-dim 128 --dtype fp16 \
    --dump-source ../fwd_d128.metal
cd .. && xcrun metal -c fwd_d128.metal -o fwd_d128.air

bash reference_air/ccv/bench_ccv/build.sh
reference_air/ccv/bench_ccv/bench_ccv_attn --d 128 --r 1024 --c 1024 \
    --dump-source reference_air/ccv/fwd_d128.metal
xcrun metal -std=metal3.1 reference_air/ccv/fwd_d128.metal \
    -o reference_air/ccv/fwd_d128.air
```

Inspect with `bash scripts/air_dis.sh <file.air>` (metal-objdump) and
diff op mixes with `uv run python scripts/air_opmix.py a.air b.air`.
