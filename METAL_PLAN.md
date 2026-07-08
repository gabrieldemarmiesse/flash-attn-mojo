# Metal flash attention in Mojo — plan & race log (branch `metal-fwd`)

Goal: FlashAttention **forward, Q/K/V only** (non-causal, dense, fp16
in / fp32 out) written in Mojo targeting Apple-silicon GPUs, **at
least as fast as** both references on this machine:

- `philipturner/metal-flash-attention` (MFA) — clone at
  `./metal-flash-attention` (gitignored)
- ccv's C++ MFA port (liuliu's extended fork: causal/GQA/batch/
  varlen/mask/sliding-window/sinks) — clone at `./ccv` (gitignored)

Then iterate feature-by-feature the way the NVIDIA/FA4 race did
(HANDOFF.md is the template). **Status 2026-07-08: milestone v1
COMPLETE — mojo is AT PARITY with ccv** (the faster reference) on
the full shape matrix. Numbers below are TRUE GPU KERNEL TIME on
both sides — the refs self-report command-buffer gpuEnd-gpuStart;
the mojo lane's is recovered from an xctrace 'Metal System Trace'
by `master_bench.py`'s default kernel-time phase (see the
measurement-protocol section):

| shape       | mojo kern µs | ccv µs | ratio | verdict |
|-------------|--------------|--------|-------|---------|
| S1024/D64   | 1162         | 1160   | 1.00x | parity |
| S1024/D128  | 2345         | 2355   | 1.00x | parity |
| S4096/D64   | 18192        | 18241  | 1.00x | parity |
| S4096/D128  | 36912        | 36882  | 1.00x | parity |
| S8192/D64   | 72837        | 72969  | 1.00x | parity |
| S8192/D128  | 154111       | ~150k  | 1.02–1.05x | parity (±5% ccv spread) |

MFA is beaten on every shape. mojo ≈ 3.46–3.77 TFLOPS = ccv's
band. The old 1.05x at S1024/D64 was PURELY the mojo lane's
~160 µs/enqueue wall-clock bias — the wall column still shows it
(1214 wall vs 1162 kernel); kernel-to-kernel it is dead-on ccv.
Correctness: max|err| ≈ 9e-6 vs the strided fp32 CPU reference
(same band as the references — f16 S-GEMM accumulate). See "The
v1 fight — resolved" below for the three mechanisms that mattered.

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
- Timing fine print: references report command-buffer GPU time
  (gpuEnd-gpuStart); the mojo lane can only wall-clock enqueue+sync
  (its only reliable in-process bracket) and so eats ~160 µs/dispatch
  of overhead the refs don't. `master_bench.py` therefore runs a
  DEFAULT-ON kernel-time phase after the wall bench: it xctrace-records
  ONLY the mojo lane (the refs already self-report GPU time) and
  recovers its true per-dispatch GPU time via
  `xctrace_gpu_intervals.steady_state_kernel_us`. That estimator is
  fragmentation-proof: a long dispatch is SPLIT across DVFS clock
  transitions into several intervals, so it buckets by clock, takes
  the highest clock the GPU settled at, and within it returns the min
  (no fragmentation — short kernels, matches the refs' min-over-
  dispatches) or the median of the tight top cluster (fragmentation —
  clean dispatches cluster at the top, fragments fall below). The
  capture is sized from the wall time to ~1.2 s of GPU work so DVFS
  actually ramps to a stable top clock (short shapes need hundreds of
  dispatches — else per-dispatch time reads a low ramp clock and
  over-reports; this is why the naive per-encoder median was wrong).
  xctrace is flaky (SIGSEGV on finalize ~1/3, 'missing template'
  export errors) — the phase retries 5x/shape and falls back to wall
  (flagged) on give-up. Disable with `--no-kernel-time`.

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

## The v1 fight — RESOLVED 2026-07-08 (parity in one day)

**Toolchain**: `mojo-compiler==1.0.0b2` (PyPI, 2026-06-18) contains
the 8x8 MMA fix — no conda nightly needed. pyproject now carries
darwin-markered pins (linux stays on 1.0.0b1 for the validated CUDA
codegen). Toy probe first (ptxas_ur_probe lesson): the intrinsic is
`llvm_intrinsic["llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
SIMD[dtype, 2]](a, b, c)` on 2-elem SIMD fragments — both the f16-
and f32-accumulate arms verified correct; the per-lane fragment map
is `matmul_8x8.mojo::_frag8_layout` == MFA's `morton_order`.

**The kernel** (bench/bench_mojo_metal.mojo): ccv-shaped — blocks
(16, 128), 2 simdgroups x 8 q rows, raw-unscaled f16-accum S, P
overwriting S in place (A-operand layout == C layout: zero-cost),
f32-accum O, 2-hop xor(1)/xor(8) softmax reductions, scale folded
once into m and the exp2 argument. Deliberate departures: ALL
fragments (Q/K/V/O) read/written directly from device memory — the
main loop has ZERO barriers and no functional threadgroup memory
(ccv stages Q+O through smem only because async_copy is its
ragged-edge clamp; its own K/V fast path is already direct).

**What actually gated perf** (the initial correct kernel was 3x
off; three mechanisms, in discovery order):

1. **The D=128 Q-cache register cliff.** Register-caching Q at
   D=128 (16 extra live f16 frags/lane) ran the NO-LOAD skeleton
   2.3x off the MMA roofline; the D=64 skeleton (Q cached, half the
   O frags) sat AT roofline. ccv's table caches Q only at D<=96 —
   now we know why. Fix: stream Q in 32-col chunks per c-block like
   ccv, but straight from device (the row stays L1-resident).
2. **1-D q-tile-major grid.** A (q_tile, head) 2-D grid was 2.5x off
   at S=8192/D=128: concurrently-resident threadgroups spread over
   16 heads = 16 desynced K/V streams = DRAM-bound. Flattening to
   ccv's 1-D q-tile-fastest ordering makes the resident wave share
   ONE head's K/V stream (SLC hits): S=8192/D=64 went 115.6 ms ->
   74.0 ms = instant parity.
3. **Residency throttle.** With zero threadgroup memory, residency
   runs far above ccv's 4 tgs/core (8 KiB alloc), and every extra
   resident threadgroup is another desynced K/V reader — the
   instantaneous working set blows past the SLC exactly when
   per-head K+V hits 4 MiB (S=8192/D=128 was the one shape still
   1.42x off). A dummy ~10.7 KiB threadgroup allocation (3 tgs/core;
   swept: 4 KiB=201 ms, 8=163, 10.7=154.5, 16=160) + an opaque
   never-taken use recovered 2618 -> 3555 GFLOPS.

**Codegen lessons (Metal/AIR, hard-won):**

- `alloca` in the mojo AIR dump is NOT a spill canary: InlineArray
  frag arrays always emit allocas + stack load/store in the IR, but
  the AGX backend compiler promotes them — a pure-MMA probe with the
  identical InlineArray accumulator pattern sustains 3.67 TFLOPS
  (probe_mma_rate). Register PRESSURE (mechanism 1) is what's real.
- Per-lane base pointers + comptime GEP offsets: precompute
  `k_head + frow (+ (c+fcol)*D)` once per c-block so every fragment
  load in the unrolled MMA nests is base + constant — no per-load
  64-bit index math (the references do 32-bit `apply_offset` math).
- Ablation probes lie under CSE: replacing MMA operands with
  CONSTANTS lets the backend collapse identical MMA chains (a "no
  K loads" arm measured 5.7x faster than physically possible).
  Synthetic operands must vary per fragment AND stay cheap.
- `barrier()` on Apple == `air.wg.barrier(2,1)` == ccv's only
  barrier; `syncwarp()` == `simdgroup_barrier(mem_none)` (unused).
- `std.math.exp2` lowers to `air.exp2.v2f32` — no fast-math needed.

**PyTorch integration (2026-07-08)**: the kernel is packaged as
`src/flash_attn_mojo/fwd_metal/` (mirrors `fwd_fa4/`: kernel.mojo,
launch.mojo, variant.mojo, _jit.py, __init__.py). `flash_attn_func`
routes MPS tensors to it (non-causal, no window/softcap,
seqlen % 128 == 0, hd 64/128, MHA/GQA); everything else falls to the
reference. Backward is the reference VJP (no Metal bwd kernel yet).
Tests: `tests/test_metal.py` (13, all green), correctness ~3e-5 vs
the fp32 reference.

**ZERO-COPY bridge (2026-07-08, SUPERSEDES the old "impossible"
note)**: the wrapper now binds torch's OWN MPS buffers, no host
round-trip. THE EARLIER CLAIM WAS WRONG — it said mojo "CANNOT bind
torch's MPS buffers (verified two ways: raw UnsafePointer AND
DeviceBuffer(owning=False))", but both tests bound `tensor.data_ptr()`,
which on MPS is the `id<MTLBuffer>` Obj-C OBJECT pointer, not the GPU
VA (Mojo dereferenced garbage), and neither made the foreign heap
resident (Mojo skips `useResource:` for buffers it didn't allocate, so
macOS-evicted heaps read zeros / drop writes — the "writes land
nowhere" symptom). Two missing pieces, both ported from the sibling
`causal-conv1d-mojo` repo (which runs this in CI): (1) `_mps.py`
extracts the real VA via the `gpuAddress` Obj-C selector
(`[MTLBuffer gpuAddress] + storage_offset`); (2) `_mps.revive_heaps`
touches each tensor with a tiny torch GPU op right before dispatch
(post-JIT-compile, via a `pre_dispatch` hook in `_jit.py`) +
`torch.mps.synchronize()`. The launcher (`launch.mojo`) wraps the five
VAs as plain device pointers and `enqueue_function`s them directly —
the mojo-owned staging buffers and the `enqueue_copy_from/to` are gone.
The ONE residual copy is an on-DEVICE cast+transpose to head-major fp16
(the kernel's fixed `head*seq*D` indexing) — bandwidth-bound on unified
memory, not a host round-trip. Measured end-to-end wall (B1 H16, per-call
sync, median): S1024/D128 9.82 ms → 4.46 ms (2.2x); S4096/D128 60.7 ms
→ 41.9 ms (1.45x; overhead over the ~36.9 ms kernel dropped from ~63%
to ~13%); S1024/D64 5.01 → 4.02 ms. Kernel AIR is byte-identical
(untouched), so ccv kernel parity is unaffected. Debug env:
`FLASH_ATTN_MOJO_MPS_REVIVE=always|off`.

**Next (v2 candidates)**: fold torch's native (B,S,H,D) strided layout
INTO the kernel (kills the residual on-device transpose above — the
last non-kernel cost); the S1024 dispatch-overhead bias (batch more
dispatches per command buffer, or a persistent-grid variant); causal;
bwd; bf16 (currently cast to fp16 in the wrapper).

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
