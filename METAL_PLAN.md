# Metal flash attention in Mojo — plan & race log (branch `metal-fwd`)

Goal: FlashAttention **forward, Q/K/V only** (non-causal, dense, fp16
in / fp32 out) written in Mojo targeting Apple-silicon GPUs, **at
least as fast as** both references on this machine:

- `philipturner/metal-flash-attention` (MFA) — clone at
  `./metal-flash-attention` (gitignored)
- ccv's C++ MFA port (liuliu's extended fork: causal/GQA/batch/
  varlen/mask/sliding-window/sinks) — clone at `./ccv` (gitignored)

Then iterate feature-by-feature the way the NVIDIA/FA4 race did
(HANDOFF.md is the template). **Status 2026-06-12: milestone M0
complete** — full 3-way harness running, v0 mojo kernel correct,
references extracted and measured. v0 is 9.4–14x BEHIND the
references (expected — see "The v1 fight" below).

## Machine / toolchain ground truth (2026-06-12, all verified)

- Apple M4 Mac mini (`Mac16,10`), 10-core GPU, 16 GB unified, actively
  cooled. macOS 26.3.2, Metal 4 (GPU family apple9 — NOT apple10/M5).
- Xcode 26 + the separate **Metal Toolchain** component
  (`xcodebuild -downloadComponent MetalToolchain`); `xcrun metal`,
  `metal-objdump`, `air-opt`, `metal-nm`, `metal-source` all work.
- Mojo: pinned `mojo-compiler==1.0.0b1` + `max-mojo-libs==26.3.0`.
  `DeviceContext()` → "Apple M4", api "metal". Target:
  `air64-apple-macosx` / `apple-m4` / `+metal3_2,+air2_7_0`.
- **Verified-working GPU primitives on 1.0.0b1/M4**: thread/block
  indices, `lane_id`, threadgroup memory via
  `stack_allocation[..., AddressSpace.SHARED]` + `barrier()`, warp
  ops (`warp.sum/shuffle_xor/shuffle_down/shuffle_idx/broadcast`,
  WARP_SIZE=32 comptime — the DeviceAttribute query throws), atomics
  (f32/i32), fp16 + SIMD[f16,8] vector loads, **bf16 arithmetic**,
  exp2/fast math, `MAX_THREADS_PER_BLOCK_METADATA` (→
  max_total_threads_per_threadgroup). Limits: 1024 threads/tg, 32 KiB
  threadgroup memory, 10 cores.
- **Missing on 1.0.0b1/M4 — the big one: NO matrix ops.** The 16x16
  simdgroup MMA compiles but Metal rejects it at pipeline creation
  (GPUFamily10 = M5+ only). The 8x8
  `air.simdgroup_matrix_8x8_multiply_accumulate` — the engine of BOTH
  references — **SIGSEGVs the 1.0.0b1 compiler**; support landed on
  modular main 2026-06-11 (commit `30f889b67f`), together with
  `linalg.matmul.gpu.apple` and `nn.attention.gpu.apple`. Nightlies:
  conda (`conda.modular.com/max-nightly`) — no PyPI nightly found.
- `async_copy` is a documented synchronous fallback on Apple GPUs
  (plain load/store; requires `address_space_cast[GLOBAL]` on buffer
  pointers). The references' `simdgroup_async_copy` AIR intrinsics
  have no mojo exposure.
- Mojo **AIR dump**: `ctx.compile_function[k, k, dump_asm=
  StaticString(path)]()` writes **textual AIR LLVM IR** (.ll;
  `dump_asm` == `dump_llvm` on metal). No env-var route (no
  MOJO_DUMP_PTX analog) — the path is comptime, baked per variant.
  `xcrun metal -c file.ll -o file.air` round-trips it to bitcode.
- Mojo dispatch overhead ~160 µs per enqueued kernel (noop floor) —
  invisible at canonical shapes (≥ ms), dominant below S≈1024
  single-head.

## The analogy map (NVIDIA race → Metal race), all verified

| NVIDIA setup                 | Metal equivalent                              |
|------------------------------|-----------------------------------------------|
| PTX dumped from mojo         | textual AIR LLVM IR via `dump_asm`            |
| `reference_ptx/` (FA4)       | `reference_air/{mfa,ccv}/` (.metal + .air)    |
| SASS op-mix diff             | `scripts/air_opmix.py` (AIR opcode histogram; |
|                              | `call air.simdgroup_matrix_8x8_*` = the HGMMA |
|                              | count). No SASS analog exists — AIR is the    |
|                              | deepest public artifact (driver JITs the ISA).|
| ptxas -v spill canary        | none headless. Closest: AIR `alloca` count /  |
|                              | op-mix regressions; Xcode GUI shader profiler |
|                              | has occupancy + limiter (not scriptable).     |
| CUPTI kernel-only time       | MTLCommandBuffer gpuStart/EndTime (== counter-|
|                              | sample timestamps exactly, 1 ns ticks;        |
|                              | atDispatchBoundary unsupported on M4 → batch  |
|                              | ≥5 dispatches/command-buffer and divide)      |
| ncu                          | `xctrace record --template 'Metal System      |
|                              | Trace'` → `scripts/xctrace_gpu_intervals.py`  |
|                              | (per-encoder GPU times, scriptable). HW       |
|                              | counters (ALU/BW/occupancy): Instruments GUI  |
|                              | only. `.gputrace` capture: producible         |
|                              | headlessly, inspectable only in Xcode.        |
| locked clocks                | none. M4 supports "consistent performance     |
|                              | state" but only the Instruments GUI can       |
|                              | enable it. Protocol: warmup ≥3, ≥5 dispatch/  |
|                              | cb, process-interleaved rounds, quote spreads |
|                              | (measured: 0.0–0.6% at S≥1024 H=16; ~3-9% for |
|                              | sub-ms single-head shapes). `sudo             |
|                              | powermetrics -s gpu_power` to audit throttle. |

## The harness (all committed, all working)

- `scripts/master_bench.py [--full] [--impls ...] [--profile IMPL]
  [--no-asm]` — auto-detects the Metal backend on darwin (quick tier
  by default; `--full` for the full shape sweep), builds the three
  lanes, runs the correctness gate (S=1024, H=16, D=64+128,
  strided-row fp32/f64 CPU references in each CLI), the interleaved
  bench matrix, refreshes `air/` from the mojo dump, prints AIR
  op-mix diffs vs `reference_air/`, and optionally wraps a run in
  xctrace.
- `scripts/bench_metal.py` — the process-interleaved orchestrator
  (per-round round-robin; pooled trials; min/median/spread/GFLOPS;
  `vs mfa` / `vs ccv` ratio columns, <1 = mojo faster, FA4-race
  convention; correctness gate fails the run loudly).
- Common CLI contract (one JSON line on stdout): mojo
  `bench/bench_mojo_metal.mojo` (built to `bench/build/`), MFA
  `reference_air/mfa/bench_mfa/`, ccv `reference_air/ccv/bench_ccv/`.
  Canonical shape: **B=1, H=16, S∈{1024,4096,8192}, D∈{64,128},
  fp16 in / fp32 out** (both references page O as fp32 by design;
  MFA's lane emulates MHA per its docs — one dispatch per head,
  per-head buffers so heads run concurrently like ccv's batched
  kernel).
- Timing fine print: references report command-buffer GPU time; the
  mojo lane reports wall-clock around enqueue+sync (its only reliable
  bracket today) and so eats its own ~160 µs/dispatch overhead — a
  deliberate conservative bias against mojo, <1% at canonical shapes.

## Reference ground truth

See `reference_air/README.md` for the committed target table
(measured on this machine, interleaved). Headline, fp16 H=16:

| S    | D   | MFA µs  | ccv µs  | mojo v0 µs | v0 vs ccv |
|------|-----|---------|---------|------------|-----------|
| 1024 | 64  | 1651    | 1161    | 11099      | 9.6x      |
| 1024 | 128 | 3270    | 2354    | 34181      | 14.5x     |
| 4096 | 64  | 19618   | 18241   | 171966     | 9.4x      |
| 4096 | 128 | 40223   | 36882   | 530799     | 14.4x     |
| 8192 | 64  | 75934   | 72967   | 687609     | 9.4x      |
| 8192 | 128 | 156242  | 149259  | 2096330    | 14.0x     |

ccv ≈ 3.65–3.77 TFLOPS, MFA 4–8% behind it, mojo v0 ≈ 250–400
GFLOPS. Correctness: mojo max|err| ≈ 3e-8 (fp32 accumulation
everywhere) vs references' ≈ 1e-5 (fp16 S-GEMM accumulate).

Both references run THE SAME engine (confirmed in AIR):
`air.simdgroup_matrix_8x8_multiply_accumulate` (v64f16 accumulate for
S=QK^T, v64f32 for O) + `air.simdgroup_async_copy_2d` threadgroup
staging, 64 threads/tg (2 simdgroups), 8 KiB tg memory, blocks
(parallelization=16, traversal=128, head-block=32) on M4/apple9.

**macOS 26 gotcha (cost a debugging session, now automated)**: Metal
4's compiler rejects `__asm("air.x.y")` labels containing dots —
breaks both MFA-family source generators. Fix: `__asm("\01air.x.y")`
(LLVM literal-symbol escape, identical AIR symbol). Auto-applied by
`bench_mfa` (runtime string substitution) and `ccv/build.sh`
(idempotent sed). Without it ccv silently falls back to a
macOS-13-era precompiled metallib — watch for `runtime_compile_ok=1`
on stderr. Mojo is immune (emits AIR directly, never MSL).

## v0 mojo kernel (`bench/bench_mojo_metal.mojo`) — DONE, correct

One threadgroup per (16 q rows × head), grid (S/16, H); 128 threads
(4 simdgroups, 4 q rows each); K/V streamed through threadgroup
memory in 32-row tiles (+8 f16 row padding); QK^T: lane j owns K row
j, SIMD8 fp16 loads, fp32 accumulate, log2-domain online softmax
(butterfly max, `warp.sum` for l); PV: per-row `shuffle_idx`
broadcast of p_j while 32 lanes split D columns; fp32 O + natural-log
LSE epilogue. Envelope: D∈{64,128}, S%32==0, fp16. Codegen notes:
comptime `sqrt` is uninterpretable on the metal target (lowers to
`llvm.air.sqrt`) — hardcode 1/√D; big-SIMD `slice`/`insert` needs
`llvm.vector.insert`, which the AIR backend lacks — use InlineArray
of per-row SIMD chunks (comptime indices → SROA keeps them in
registers).

## The v1 fight (next milestone — making mojo fast)

The op-mix diff says it all: references = 3 MMA calls + async copies
in the hot loop; mojo v0 = 0 matrix ops, 128 `air.simd_shuffle` (PV
broadcasts), ~3000 scalar-vector ops. The path:

1. **Toolchain bump** to a nightly with the 8x8 MMA fix (conda-only;
   decide: side venv/conda env just for the metal kernels vs waiting
   for the next stable). Probe the 8x8 MMA with a toy GEMM FIRST
   (ptxas_ur_probe lesson: test codegen hypotheses on toys).
2. Rebuild the kernel MFA-shaped: 8x8 simdgroup-matrix tiles,
   register-resident O accumulator, 2 simdgroups/tg, async-copy
   staging if/when exposed (or plain loads — M4 has no real async
   copy in mojo anyway; MFA still hits 3.5 TFLOPS with it, so the
   copies are not the moat — the MMAs are).
3. Iterate with `master_bench.py` + AIR op-mix diff per edit,
   exactly like the PTX workflow. Watch `alloca` in the mojo AIR as
   the spill-canary stand-in.

Fallback if the nightly bump is blocked: vendor the 8x8 MMA as a
`llvm_intrinsic` call on `<64 x half>` SIMD values from 1.0.0b1 —
risky (that intrinsic path is what SIGSEGVs), so toy-probe first.

## Milestone M0 checklist (this ping) — ALL DONE

1. ☑ branch `metal-fwd` from latest main
2. ☑ plan drafted and filled (this file)
3. ☑ `reference_air/` populated (MSL + AIR + README + targets)
4. ☑ both reference CLIs build & run with JSON timing output
5. ☑ mojo v0 fwd kernel correct (3e-8) at all canonical shapes
6. ☑ `master_bench.py` end-to-end: gate + interleaved 3-way
   table + AIR dump + op-mix diff + `--profile` xctrace hook
7. ☑ profiling-tools writeup (analogy-map table above)

Out of scope for M0 (deliberately): making mojo fast (v1), causal,
backward, bf16, varlen, torch/`flash_attn_func` integration on Mac.
