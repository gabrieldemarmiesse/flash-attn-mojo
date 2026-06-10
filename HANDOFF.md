# flash_attn_mojo — handoff notes

Active project: **match Tri Dao FlashAttention-4 on this H100** with
a minimalist Mojo fwd kernel. See CLAUDE.md "Current focus" for the
full picture; this file tracks session-to-session state.

## State as of 2026-06-10

`src/flash_attn_mojo/fwd_fa4/` v7 — correct, **~1.02-1.05x FA4
kernel time** at the canonical config (bf16, hdim128, non-causal,
B=2 S=8192 H=16). FA4 ~2240-2350 us / ~490 TFLOPS, mojo ~2280-2440
us / ~480 TFLOPS (both drift with sustained-load clocks; always
bench A/B interleaved — `scripts/master_bench.sh`).

Architecture (mirrors FA4's sm90 fwd): 384 threads = producer
warpgroup (thread 0 issues all TMA; setmaxnreg 24) + 2 MMA
warpgroups (setmaxnreg 240, 168 regs/thread actual); K/V in a
6-slot smem ring with full/empty mbarrier pairs; FA4's
intra-warpgroup overlap schedule (QK(n+1)+PV(n) committed
back-to-back, wait_group(1), softmax overlapping PV(n),
wait_group(0)); pingpong named barriers 1/2 alternating the two MMA
warpgroups' issue phases; epilogue stages O in smem (16B-chunk-major
for the unswizzled descriptor) and TMA bulk-stores.

Perf history at the canonical shape (kernel-only CUPTI us):
v1 single-WG-pair 4004 → v2 2-stage pipeline 3222 → v3 overlap
schedule 3035 → v4 warp specialization 2500 → v5 TMA-store epilogue
~2400 → v6 pingpong 2355 → v7 incremental ring state ~2280-2340.

## What moved the needle (ncu lessons)

- Block-wide `barrier()` per loop iter: stall 1.41/issue → warp
  specialization with full/empty mbarriers killed it.
- Per-iteration `%6` / `//3` slot math: +17% ALU instructions vs
  FA4 → incremental wrap-around counters (~3% total time).
- Tree-reduction softmax (4 partial accumulators): made it *slower*
  (2355 → 2440); reverted. ptxas seems to handle the serial chain
  better than the extra live registers.

## Backward (bwd_fa4) — state as of 2026-06-10 (fourth session)

Correct (bf16 noise floor vs fp32 autograd at S in {128, 256, 640,
1024}: dq/dk/dv 1.4-1.8e-3) at **~1.22x FA4** with LOCKED clocks
(`sudo nvidia-smi --lock-gpu-clocks=1500,1500` — DO THIS FIRST,
unlocked clocks drift ±4-5% run-to-run and drown <3% experiments;
persistence mode is off so relock per session). Main kernel
7.42-7.56 ms vs FA4 6.05-6.08; preprocess (167 vs 159 us) and
convert (115 vs 109 us) at parity. Loop:
`scripts/master_bench.sh --kind bwd`; fast check:
`scripts/bench_fa4.py --impl mojo --kind bwd --check-only` (covers
tail (S%%80!=0) and exact-fit seqlens).

Perf journey: 141.8 (acquire atomics) -> 28.5 (relaxed) -> 24.3
(manual sdS addressing) -> 11.5 (balanced hand-rolled dQ^T) -> 10.6
(FA4 commit order) -> 9.3 (dQ smem mailbox + cp.reduce drain) ->
8.6 (local-memory fix + convert rewrite) -> 8.0-8.1 total
(**fourth session: tile_m=80 port + FA4 wait order + producer bulk
lse/dps + bf16-P dS + stmatrix epilogue**).

Architecture now matches FA4's structurally (verified against the
reference PTX by a 4-reader audit; per-iter wgmma inventory is
identical: 24x m64n80k16 + 10x m64n128k16):
- tile_m=80: BM=80 everywhere; side buffers padded to
  Spad=ceil(S/80)*80 with lse=+inf / dpsum=0 pad rows (preprocess
  writes them; the tail m-tile's P and dS become exact zeros, so
  TMA OOB zero-fill / finite next-batch garbage reads are
  annihilated). Convert kernel: one CTA per 80-row m-block,
  [wg(2)][chunk(10)][tid(128)][4] decode + row predicate.
- Q/dO ring: 4 slots (2 stages each — the smem cap forces it and
  it is FA4's config). smem total 230912 B of 232448.
- NO stager warp: producer warp 0 issues, per stage, the TMA copy
  plus a 320-B 1-D `cp.async.bulk` of lse (Q slot) / dpsum (dO
  slot) onto the same mbarrier (expect_tx 20800). Inlined asm
  ("r,l,r,r"); stdlib has no wrapper.
- sdS: FA4's k-major (80 q-rows, 128 kv-cols) SW128 tile, written
  by 5x `st_matrix[simd_width=4, transpose=True]` per thread (each
  call: 16 q-rows x the warp's 16 kv-cols, in the wg's own 64-col
  slab; data = c-frag bf16 pairs 8i..8i+7 in order). Swizzle XOR
  from the ABSOLUTE address: `addr ^ ((addr>>3)&112)` — the
  dynamic-smem base phase folds in for free. The old mn-major
  pair-store scheme is impossible at BM=80 (needs 64-elem rows).
- dQ^T GEMM: hand-rolled m64n80k16 SS, layout_a="col" (trans-a=1),
  layout_b="col" (trans-b=0, k-major sdS — FA4's imm tail 1,0).
  B k-steps are two-level: +32 B x4 in-slab, +10240 B across.
  c-regs via StaticTuple[Float32, 40] (SIMD[f32,40] is a comptime
  assert failure) copied to dq_reg after issue (mem2reg aliases).
- FA4's wait order {1,0,1,1,0}: dP retired (wait 0) BEFORE the dS
  math; P and dS packed bf16 together after; dV committed after
  the dS store. This is REGISTER-PRESSURE structure, not overlap:
  it keeps the f32 P and bf16 P from coexisting at the peak.
  Additionally dS is computed from the bf16 P (the same rounding
  dV consumes) freeing the 40 f32 P regs entirely.
- Epilogue: dV/dK staged via 8x st_matrix x4 (non-trans) into the
  dead K/V SW128 tiles and stored with 2 big SWIZZLE_128B TMA
  copies each, dV store overlapping dK staging. NOTE: here the
  swizzle XOR must be applied PER CALL (the 32-B column steps live
  in the swizzled bits; the dS store's 2048-B steps dodge this).

### The remaining ~22%: a compiler-level wall

ncu PC sampling side-by-side (locked clocks): tensor pipe 55 vs
78%, issue_active 18 vs 26%; the ONLY large stall delta is
long_scoreboard 5.3 vs 2.3 cyc/issue = consumer LOCAL-memory spill
reloads. ptxas spills ~160-200 B/thread (`ptxas -v` on the dumped
PTX shows it; nvdisasm places the LDL/STL inside the consumer hot
loop). Root cause: FA4/cutlass keeps descriptors, smem addresses
and loop counters on the UNIFORM datapath (UR registers — a
separate 63-reg/warp file); mojo+LLVM materializes them in regular
registers (R2UR storm, ~11%+7% IMAD.MOV of stall samples, plus a
per-iter S2R SR_TID.X remat), and at the 240-reg cliff (208 f32 of
live accumulator/fragment data is irreducible) the addressing
state spills. EIGHT source-level attacks failed (see negative
results). Realistic paths to parity: (a) mojo/LLVM gaining UR
allocation, (b) post-processing the PTX before ptxas (no hook in
compile_function today), (c) finding ~30 regular registers some
other way nobody has thought of yet.

RULED OUT — ptxas version (checked 2026-06-10): mojo compiles
in-process via the statically linked `modular/lib/libNVPTX.so`
(ptxas 13.1.115; overridable with `MODULAR_NVPTX_COMPILER_PATH` —
our `__init__.py` auto-points it at the `nvidia-cuda-nvcc-cu12`
wheel if installed). FA4's cute DSL compiles via its own embedded
ptxas 12.9.83 inside `_cutlass_ir...so` (the installed lib is the
libs_base CUDA-12 variant; cubins load via cuModuleLoadData — so
NEITHER side uses the driver JIT; cubin ELF ABI-version 7 vs 8
confirms the 12.x/13.x split, and `CUTE_DSL_KEEP=cubin` dumps
FA4's cubins for inspection). Rebuilding our kernel with ptxas
12.9.86 (FA4's generation) produced BYTE-IDENTICAL SASS to the
13.1.115 build — same 55 STL/LDL, same 77 R2UR, same 1608-instr
mix. The spills/non-uniformization are a property of our PTX's
dataflow shape, not the ptxas version. Also noted: mojo emits
`.version 8.5` PTX vs FA4's 8.8 — no observed consequence.

### Diagnosis methodology that worked

- LOCK THE CLOCKS before measuring anything (see above).
- 4-reader audit workflow (FA4 cute source / FA4 reference PTX /
  our kernel / stdlib capabilities -> synthesized port spec with a
  delta table) made the tile_m=80 port land CORRECT ON FIRST
  COMPILE — including stmatrix address math copied verbatim from
  FA4's PTX decomposition (lines 716-781).
- `ptxas -arch=sm_90a -v ptx/mojo_bwd_fa4.ptx` after every build:
  the spill-bytes line is the canary. nvdisasm -c to see where.
- PROBE_NO_* comptime flags for bubble attribution; ncu PC sampling
  (SourceCounters -> --page source --csv) for stall attribution;
  sector tables for coalescing.
- ptxas honors setmaxnreg: pool 2*128*240 + 128*24 = 384*168.

### Negative results (don't retry blindly)

Fourth session (all measured interleaved, most pre-clock-lock):
- Consumer m-loop unrolled by the ring period (comptime slot/stage)
  TRIPLED spills (616 B) and cost +1.7 ms: both bodies' liveness
  overlaps at the seam. Same cliff mechanics as descriptor
  hoisting.
- Hoisting the (loop-invariant!) sdS swizzle offset + mailbox
  pointer out of the m-loop: spills 164->156 B but slower — ptxas
  in-loop remat was already optimal.
- Fusing the dS-math walk with the P/dS packing walks per-cc:
  spills 196 B, +600 us.
- dK as SS (A = sdS slab from smem, killing ds_reg's 20 regs):
  1.31x — the smem round-trip loses to the RS fast path.
- Mailbox reading dq_tup directly with per-element "+f" fences
  (instead of the aliased dq_reg copy): neutral-to-negative.
- Elected per-warp empty arrives (FA4's convention, count 8 vs
  256): reproducibly ~2-4% SLOWER in our codegen.
Third session and earlier:
- Cross-iteration software pipelining: 11.3 -> 15.4 ms.
- Descriptor/pointer hoisting: 8.2 -> 8.9 ms.
- Reg split 40/232: 8.2 -> 8.5 ms.
- L2 cache hints: FA4 passes ZERO policy handles — a no-op.
- LLVM CSEs adjacent identical wgmma.wait_group intrinsics and
  floats mbarrier arrives past wgmma asm.

Hard-won bugs (see also memory notes):
- smem swizzle XOR uses the ABSOLUTE address; fold the base phase
  (or compute `addr ^ ((addr>>3)&112)` on the absolute address).
  For strided stmatrix calls, re-apply per call unless the step
  provably avoids bits 4-6 (2048-B steps do; 32-B steps do not).
- Epilogue staging overwrites K/V smem under the other wg's last
  dQ GEMM -> pre-epilogue named barrier.
- wgmma trans bits in PTX are ground truth: FA4 bwd dQ =
  (trans-a=1, trans-b=0); mojo layout strings map A:"col"->1,
  B:"col"->0, B:"row"->1.
- The dq_tup -> dq_reg copy sits between wgmma issue and
  wait_group; it is sound ONLY because mem2reg aliases it (no real
  MOVs). If grads go wrong after touching that code, check the
  SASS for real register moves before the GMMA scoreboard clears.

## Not yet tried (fwd)

- ~~L2 cache hints on TMA loads~~ DEBUNKED (third session): FA4's
  cache_hint operands are all zero policy handles — no-op.
- 32-bit (`layout_int_type=DType.int32`) LayoutTensor index types
  (PTX still has ~50 add.s64 vs FA4's 0).
- Profile-guided softmax reordering with SASS (`--set source`).
- Locking clocks for stable A/B (`sudo nvidia-smi -lgc <freq>`).

## Legacy

The FA2-targeting `fwd/`, `fwd_fa3/`, `bwd/` subpackages and their
tests/benches predate the FA4 race and are throwaway-grade reference
(user call). Clean up once FA4 parity is declared.
