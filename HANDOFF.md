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

**AT PARITY WITH FA4** (within FA4's own run-to-run variance):
locked-clock interleaved master_bench x3 -> main kernel
6148/6172/6250 us vs FA4 5913/6053/6176 = ratio 1.000/1.023/1.059.
Loop body: 448 instructions/iteration vs FA4's 532 (identical 34
HGMMA of tensor work) — we now run FEWER instructions per iteration
(fma softmax vs mul+sub, no seqlen masks). Correct at bf16 noise
floor (dq/dk/dv 1.43/1.33/1.42e-3 at S=1024; tail + exact-fit
seqlens covered by --check-only). ALWAYS
`sudo nvidia-smi --lock-gpu-clocks=1500,1500` before measuring
(persistence off — relock per session); even locked, both kernels
wobble ~2-4% run-to-run, so quote 3-run spreads.

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

### The codegen wall: found and broken (was ~22%, now ~10%)

The spill mechanism was isolated by a PTX bisection harness
(`scripts/ptxas_ur_probe.py` — generates toy wgmma-loop PTX in
varying dataflow shapes, compiles with ptxas, reads back UR
allocation / R2UR / spills) plus a 2-reader PTX anatomy audit:

- ptxas's warp-uniformity analysis was FINE on our PTX: ring
  counters (even 64-bit selp-wrapped ones), in-loop-recomputed
  B-descriptors, mbarrier addresses, the magic div-by-80 — all
  landed in URs. tid>>7 warpgroup indexing, setmaxnreg, mbarrier
  spins, selp rings, 64-bit imm chains: all uniformity-safe in
  isolation (probe-verified).
- The killer was CAPACITY: LLVM hoisted 24 loop-invariant 64-bit
  A-descriptor k-step variants (S^T/dP^T/dQ^T x 8) into the loop
  preheader. At ~30 URs already in use they overflow the 63-UR/warp
  uniform file; ptxas spilled them to LOCAL (164 B) and reloaded +
  R2UR'd before each HGMMA -> long_scoreboard 5.3 cyc/issue.
  Probe repro: >=16 live 64-bit descriptors hits the UR cliff;
  many24 + 180 live f32 = 108 spill B + 40 R2UR (the kernel's exact
  pathology); 32-bit-root rematerialization = 0/0 at any count.
- FA4's cute PTX REBUILDS every descriptor every iteration from
  32-bit `mov.u32 r, <shared symbol>` roots (`.pragma "nounroll"`,
  3 loop-carried 32-bit scalars total, warp roles made provably
  uniform via `shfl.sync.idx` lane-0 broadcast). Rematerialization
  on the uniform datapath is free; LIVENESS is what kills.

Fixes shipped (kernel.mojo, consumer loop top):
1. Launder the K/V tile pointers through a no-op `mov.b32` inline
   asm (+ a `warp.broadcast` lane-0 shfl, FA4's uniformity idiom)
   so LLVM cannot hoist the descriptor variants; they are rebuilt
   per iteration and ptxas folds them into UIADD3/ULOP3 immediates.
2. `(x - y) // 2` on Int = SIGNED floor-div = a 17-op rounding
   correction chain per smem address (10 stmatrix sites). Replaced
   with `(x >> 1) - (y >> 1)`.

Result: ptxas spills 164 -> 0 B; long_scoreboard 5.33 -> 1.87.
That left ~10% of instruction-mix gap, closed by the SASS op-mix
diff (extract both innermost loop bodies — same 34 HGMMA — and
diff opcode histograms; started at 638 vs 532 instr/iter):

- THE TID-WIDENING TRAP (the 52-R2UR storm's root): mojo's
  `Int(thread_idx.x) // 128` widens tid to 64-bit BEFORE the
  shift. ptxas's tid-uniformity rule only matches 32-bit shr.u32
  (probe: tid7 = 0 R2UR vs tid7w = 32 R2UR), so EVERY wg-derived
  value (per-wg A-desc offsets, mailbox base, barrier ids) was
  per-thread. LLVM re-canonicalizes 32-bit extracts of widened tid
  back to shr.u64; the robust fix is
  `warp.broadcast(Int32(tid >> 7))` — convergent so LLVM keeps it,
  a recognized broadcast so ptxas uniformizes. 638 -> 532
  instr/iter, R2UR 52 -> 3 in one line.
- One launder shfl, not two: V's root = K's + tile size (shfl is
  MIO; mio_throttle was 1.28 vs FA4 0.71).
- bf16-P dS reverted (register headroom exists post-spill-fix):
  -40 F2F on the dS critical path, -20 PRMT, precision back to
  1.4e-3. 532 -> 448 instr/iter.

Remaining structural diffs vs FA4 are now in OUR favor or neutral;
the ~0-5% residual is within measurement noise. If more is ever
needed: barrier 1.71 vs 1.45 and wait 1.43 vs 0.95 stalls are the
only deltas left (wgmma wait placement / 2-stage prefetch depth).

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

## Varlen (2026-06-11, fifth session)

Packed cu_seqlens support across all four kernels (fwd + the bwd
trio), `flash_attn_varlen_func` differentiable end-to-end. v1
envelope: every seqlen % 128 == 0, self-attn lengths, bwd MHA-only.

Results at the canonical packed config (8 seqs, 16384 tokens,
lengths 1280–3072, H=16, locked clocks, 3-run spreads):
- bwd: 0.981–0.987x (mojo FASTER) noncausal, 1.001–1.004x causal —
  parity on the first bench, zero perf work needed.
- fwd: 1.040–1.045x both causal and noncausal — but this is NOT a
  varlen gap: degenerate-uniform cu_seqlens=[0,8192,16384] runs
  0.983–0.988x (mojo FASTER than FA4's varlen kernel), and dense
  B=8 S=2048 shows the same 1.054x. It is a pre-existing
  short-sequence amortization gap in the dense fwd (per-CTA
  trip-count ~16 vs 64); open task. Note FA4's varlen fwd is itself
  ~1–3% slower than FA4's dense fwd at identical work (its
  in-kernel scan); ours adds ~0.

Design (full spec in the audit output; deltas vs FA4):
- Host work-item tables instead of FA4's in-kernel 32-lane
  prefix-scan scheduler (~186 PTX lines/CTA): int32[8] rows, built
  with vectorized torch on a host copy of cu_seqlens (one D2H sync
  per call). fwd/preprocess/convert tables are per q-tile
  (m_local, q_row_base, seqlen_q[, mpad_base]); bwd main per
  kv-tile (n_block, q_base, k_base, slq, slk, num_m, m_start,
  mpad_base — m_start and the //80-derived fields HOST-precomputed,
  so no signed-div chains on device).
- KERNEL SIGNATURES ARE UNCHANGED vs dense (byte-identity gate per
  edit): the table address rides comptime-dead arg slots — fwd:
  sched_swizzle (LPT is dense-causal-only) with seq_len carrying
  total_q; bwd main: dk_accum_ptr (GQA varlen deferred) with
  seq_len carrying num_mpad; preprocess: dk_accum_ptr +
  seq_len=total_q + nheads=total_qpad; convert: seq_len=table addr
  + nheads=num_mpad (H from grid_dim.y).
- Loads/stores: flat TMA descriptors over packed (total, H, D),
  runtime per-seq row coords (rows=total_q / total_k at descriptor
  creation). Tile-aligned v1 → full-tile TMA stores stay safe.
- Stats: FA4's padded-packed layout verbatim — (H, total_qpad)
  with padded_offset_q[i] = ((cu_q[i]+i*BM)//BM)*BM and
  total_qpad = ceil((total_q+(nseq+1)*BM)/BM)*BM (multiplier is
  len(cu_seqlens) — under-allocating overlaps the last window;
  asserted host-side). Pad rows keep OUR +inf-LSE/0-dpsum
  convention (FA4 uses lse=0 + an in-loop seqlen_q mask — the two
  are coupled; swapping one without the other silently corrupts
  dQ).
- LSE is packed (H, total_q) (FA4's varlen layout) — the bwd
  preprocess reads it back with the same indexing.
- Every per-CTA table scalar is warp.broadcast-laundered (the
  tid-widening hazard class; spills stayed 0/168 regs across all
  varlen variants).

Bench plumbing: `bench_fa4.py --varlen [--varlen-lens ...]`
(check sets + canonical mixed bench; per-seq fp32 references; FA4
varlen cross-checked through the same harness), `master_bench.sh
--varlen` (PTX refs `reference_ptx/fa4_*_varlen.ptx`).

## The O-store epilogue fix (2026-06-11) — short-seq fwd parity

Symptom: fwd trailed FA4 by a near-constant ~35-55 µs at any
2048-CTA config — invisible at S=8192 (1.01-1.03x) but 1.05x at
S=2048 and 1.22x at S=512. Single-wave isolation (B=1 S=512, 64
CTAs): +2.5 µs per CTA flat.

Attribution (ncu --section SourceCounters PC sampling at the
single-wave shape): ~8% of all warp-stall samples sat on 16
serialized UTMASTG.3D issues + UTMACMDFLUSH — the O store. The old
design staged O row-major in smem and stored through an UNSWIZZLED
descriptor with desc_shape (BM, 1, 8): TMA decomposes that into 16
separate 16B-chunk column stores, each a ~70-cycle uniform-datapath
issue on one thread, plus the EXIT-stall drain while 11 other warps
wait (13% of samples vs FA4's 6.7%). FA4 stores the tile in 2
swizzled calls.

Fix: stage O with 8x stmatrix.x4 (non-trans) into the dead Q tile
in its native SW128 k-major layout (the bwd dK/dV epilogue pattern,
copied nearly verbatim — same m64n128 c-frag geometry, same
per-call absolute-address swizzle XOR), switch the O descriptor to
SWIZZLE_128B, and issue ONE whole-tile async_store_3d.

Results (locked clocks, interleaved): B=1 S=512 1.221x -> 1.027x;
B=8 S=2048 1.056x -> 1.004-1.006x; CANONICAL B=2 S=8192 1.025x ->
0.991-0.992x (mojo FASTER — the fix also recovered the long-S
epilogue cost, which was previously ~half the 1.00-1.03x band);
causal canonical 0.986x; varlen mixed 1.042x -> 0.997-1.001x
(0.978-0.981x causal). Residual: B=32 S=512 still ~1.05x (4-trip
CTAs, 36 waves; remaining per-CTA delta ~0.3 µs — ring-fill ramp /
upfront empty-arrives are the suspects if ever chased).

Moral: at short per-CTA trip counts, FIXED per-CTA costs dominate
the ratio, and TMA *issue* count (not bytes) is the unit of cost —
audit every async_store/copy for descriptor chunking.

## Arbitrary sequence lengths (2026-06-11, sixth session)

Lifted the varlen seqlen%128 restriction; every config stays at
parity (aligned mixed fwd 1.002-1.003x / bwd 0.997-1.003x; ragged
mixed fwd 1.002x / causal 0.980x / bwd 0.997x both).

Design (survived a 35-agent adversarial review with zero real
findings):
- fwd non-causal: ragged kv columns masked in softmax_block on the
  boundary tile ONLY — the kv walk is REVERSED under varlen
  non-causal so the boundary tile is processed in the consumer
  prologue and the steady loop stays mask-free. The first version
  branched per-iteration instead and cost a consistent 3.5%
  (1.034x): a single predicated block in the softmax hot path is
  measurable — FA4's PTX rule ("steady loop setp count identical to
  dense") is the real parity constraint. Online softmax is
  order-independent, so reversal is free.
- fwd causal: NO kv mask needed — with BM == BN == 128 and
  self-attn lengths the sequence's last kv tile is only ever
  processed as the last m-block's diagonal tile, where col > row
  already kills every garbage column for stored rows. (PROVED, and
  re-verified by the review panel.)
- bwd: S^T kv-ROW mask every m-trip on boundary CTAs — causality
  does NOT subsume here (q below the diagonal attends everything
  earlier, including garbage slots). q-side ragged tails need
  nothing: the per-seq +inf-LSE/0-dpsum padded stats windows
  already annihilate partial m-tiles.
- Partial tail tiles bypass smem staging + TMA and store c-frags
  straight to gmem row-predicated (cross-seq overwrite is the top
  hazard; the trailing-partial test set turns an overshoot into an
  OOB). Raw pointers ride free slots: O = sched_num_hb_q (LPT is
  dense-causal-only); dk/dv = an aux row appended to the kv work
  table (two int64s at row index grid_dim.x).
- Hosts: kv tile counts go ceil; envelope drops to lengths >= 1
  (zero-length rejected at the API).

Kernel signatures still byte-identical to dense (gate: 0 diff
lines), 0 spills / 168 regs on all varlen variants.

## Varlen GQA (2026-06-11, sixth session)

fwd: free (0.988-0.989x, mojo faster — pack-GQA addressing was
already in). bwd: the dense fp32-accum design carried over with a
SIMPLIFICATION over FA4: no per-seq padded accumulator windows.
The ragged S^T mask makes garbage kv rows' dV/dK c-frags EXACTLY
zero (exp2 underflow -> P=0 -> wgmma accumulates zero rows), so a
boundary CTA's full-tile cp.reduce.add into the next sequence's
rows is a numeric no-op, and concurrent atomic adds commute — only
the buffer END pads to a full tile (total_k_alloc). Accumulator
ptrs + total_k_alloc ride aux rows appended to BOTH work tables
(read at index grid_dim.x); preprocess zeroes the accumulators.

bwd result: 1.036-1.044x plain / 1.072x causal — the only config
above wobble. Per-kernel decomposition of the ~+90 us: main +50
(GQA epilogue fixed cost ~3 us/CTA x short varlen m-sweeps — the
same short-work amortization story as the fwd O-store fix, but the
cost here is the 2x serialized 64 KiB stage+reduce, already
read-overlapped), preprocess +48 (accumulator zeroing at 1.4 TB/s),
torch permute-cast +43, convert -53 (we win). FA4 dodges the
per-CTA epilogue at varlen-GQA by restructuring (pack_gqa/CLC).
Levers if parity is required: pack-GQA M-dim head packing (a tile
geometry change), a fused dkv-convert kernel (~-35 us), faster
zeroing (~-12 us).

Negative results: (a) cast-then-permute conversion split (slice
cast ran 0.9 TB/s, net SLOWER than the fused permute-cast);
(b) keeping epilogue addresses (aux ptrs, kv_row) live across the
main loop — 8-16 B spills; fixed by REBUILDING them from the table
+ special regs inside the thread-128 issue branch (the
rematerialization lesson again, host-data edition). The
cp_async_bulk_wait_group stdlib default is already .read — no win
available there.

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
