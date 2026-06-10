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

## Backward (bwd_fa4) — state as of 2026-06-10 (third session)

Correct (bf16 noise floor vs fp32 autograd) at **~8.6 ms vs FA4's
~6.4 ms total (~1.34x)**; main kernel 8.2-8.3 vs 5.9-6.2 ms, the
preprocess (167 vs 159 us) and convert (114 vs 109 us) kernels are
at FA4 parity. Loop: `scripts/master_bench.sh --kind bwd`; fast
check: `scripts/bench_fa4.py --impl mojo --kind bwd --check-only`
(bench prints per-kernel `KERNEL` lines now).

Perf journey: 141.8 (acquire atomics) -> 28.5 (relaxed) -> 24.3
(manual sdS addressing, red.v2 inline asm) -> 11.5 (v3: balanced
hand-rolled dQ^T over both warpgroups) -> 10.6 (v4: FA4 commit order
+ producer-staged lse/dpsum) -> 9.3 (v5: dQ smem mailbox +
cp.reduce.async.bulk drain warp) -> 8.6 ms (lse/dps local-memory fix
+ convert rewrite).

Architecture mirrors FA4's mma_one_m_block: S^T/dP^T swapAB wgmma
(wait_group(1) tricks), P/dS in registers, dV/dK RS wgmma, dS^T
staged to double-buffered swizzled smem read by a HAND-ROLLED
dQ^T = K^T·dS wgmma (`wgmma_async[layout_a="col"]`, mn-major A
descriptor — TensorCoreAsync can't express it). dQ c-frags are
dumped raw (8x st.shared.v4) into a per-wg 16 KiB smem mailbox that
producer warp 1 drains with `cp.reduce.async.bulk...add.f32`
(inlined asm — the stdlib wrapper is wrongly gated SM100+) into
dq_accum, an OPAQUE blocked fragment dump (per m-block:
[wg(2)][chunk(8)][tid(128)][4] f32 contiguous) that the convert
kernel decodes. Mailbox protocol = FA4's: named barriers 6/7 (full)
9/10 (empty), count 160, gated by cp.async.bulk.wait_group.read.
Producer warp 0 = tight TMA issue only; warp 1 = dQ drain; warp 2 =
lse/dpsum stager (rides the Q slot's full barrier, init(2)).
dK/dV TMA-stored from reused K/V smem.

### Diagnosis methodology that worked (third session)

- `PROBE_NO_*` comptime flags in kernel.mojo compile out subsystems
  to attribute bubbles. Result: dQ subsystem = ~3.1ms of which
  mailbox handoff ~2.2ms; cp.reduce itself and the proxy fence are
  nearly free; the dQ GEMM fully overlaps.
- ncu PC sampling (`--section SourceCounters`, then `--page source
  --print-source sass --csv`, sort by 'Warp Stall Sampling') beats
  metric-level stalls. Found: (1) stack_allocation arrays live in
  LOCAL memory (LDL/STL = long_scoreboard) — fixed by loading
  lse/dps pairs from smem at use sites; (2) R2UR+IMAD.MOV descriptor
  plumbing = 25-30% of stall samples vs FA4's ~0% (cutlass keeps
  descriptors on the uniform datapath, UIADD3/ULEA/UMOV). This is
  the main remaining instruction-level delta.
- ncu sector tables (l1tex__t_sectors per request) found convert's
  16x write amplification (16B/lane scattered -> 32 half-sectors per
  warp store).
- ptxas DOES honor setmaxnreg: consumer SASS uses R232+ (240 cap);
  the pool math 2*128*240 + 128*24 = 384*168 is exact.

### Negative results (don't retry blindly)

- Cross-iteration software pipelining (commit S/dP(n+1) at iter end)
  REGRESSED 11.3 -> 15.4 ms.
- Hoisting descriptors/pointers out of the consumer loop REGRESSED
  8.2 -> 8.9 ms (cross-loop liveness costs more than remat at this
  register pressure).
- Producer/consumer reg split 40/232 REGRESSED 8.2 -> 8.5 ms
  (consumers are the reg-starved side).
- L2 cache hints: FA4's `.L2::cache_hint` instructions all pass a
  ZERO policy handle (cute always emits the hint form) — not a real
  difference, nothing to copy. (Also applies to the fwd ideas list.)
- LLVM CSEs adjacent identical `wgmma.wait_group` intrinsics and
  reorders mbarrier arrives past wgmma asm: the intended
  "wait(1)->release dO->commit dK->wait(1)" collapses to one wait
  after dK. Harmless here but don't trust source-order scheduling.

Hard-won bugs (see also memory notes):
- smem swizzle XOR uses the ABSOLUTE address ((addr>>7)&7); dynamic
  smem is phase-shifted by static smem (mbarriers) -> hand-rolled
  swizzled stores must fold `(Int(base)>>7)&7` into the XOR. All
  TMA-fed operands are immune (encode/decode both absolute) — an
  aligned standalone unit test of the GEMM also passes. Debug via
  in-kernel smem dumps to dq_accum.
- Epilogue dK/dV staging overwrites K/V smem under the other wg's
  in-flight last dQ GEMM -> pre-epilogue named barrier.
- wgmma trans bits in PTX are the ground truth for operand
  major-ness: FA4 bwd dQ = (trans-a=1, trans-b=0); layout strings
  map A:"col"->1, B:"row"->1.

Negative result: a cross-iteration software pipeline (commit
S/dP(n+1) at iteration end, drain dQ(n-1) at iteration top)
REGRESSED 11.3 -> 15.4 ms with identical wgmma order, no spills,
same 168 regs. Top-of-iteration commits win; don't retry blindly.

Current profile (ncu, main kernel): tensor pipe ~48% busy (FA4
78%), SM throughput 48% vs 78% at identical occupancy/regs/launch.
Remaining stall budget: spin branches+VOTEU ~18% (steady-state
producer/consumer idle, FA4 pays ~20% too), R2UR+IMAD.MOV descriptor
plumbing ~25% (FA4 ~2%: uniform datapath), FENCE.VIEW 3.3% (FA4
6.1%). Next levers:
- tile_m=80 like FA4 (THE structural delta: 103 vs 128 outer
  iterations amortize all per-iter fixed costs; needs n80 wgmma,
  k-major dS smem with trans-b=0, TMA OOB-zero tail handling for
  S %% 80 != 0, padded lse smem, 40-elem c-frags).
- Find what blocks ptxas from uniformizing our descriptor math
  (FA4's SASS does UIADD3/ULEA on URs; ours rematerializes via
  R2UR). In-loop recompute is currently better than hoisting; the
  blocker is likely the LLVM-level dataflow shape, not the math.
- Convert kernel is at parity; preprocess at parity.

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
