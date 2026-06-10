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

## Backward (bwd_fa4) — state as of 2026-06-10 (second session)

Correct (bf16 noise floor vs fp32 autograd, 2.4e-4 vs FA4 grads) at
**10.6 ms vs FA4's 6.6 ms (1.61x)**. Loop: `scripts/master_bench.sh
--kind bwd`; fast check: `scripts/bench_fa4.py --impl mojo --kind
bwd --check-only`.

Perf journey: 141.8 (acquire atomics) -> 28.5 (relaxed) -> 24.3
(manual sdS addressing, coalesced pre/convert, red.v2 inline asm,
deferred waits) -> 11.5 (v3: balanced hand-rolled dQ^T over both
warpgroups) -> 10.6 ms (v4: FA4's dV,dQ,dK commit order so the dQ
drain overlaps the dK GEMM + lse/dpsum staged by the producer warp
riding the Q pipeline, full[Qslot].init(2)).

Architecture mirrors FA4's mma_one_m_block: S^T/dP^T swapAB wgmma
(wait_group(1) tricks), P/dS in registers, dV/dK RS wgmma, dS^T
staged to double-buffered swizzled smem read by a HAND-ROLLED
dQ^T = K^T·dS wgmma (`wgmma_async[layout_a="col"]`, mn-major A
descriptor — TensorCoreAsync can't express it), dq_accum (B,H,D,S)
fp32 via red.relaxed.v2.f32, dK/dV TMA-stored from reused K/V smem.

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

Current profile (ncu, main kernel): tensor pipe 41.5% busy (FA4
~53%), cycles 14.2M, stalls: long_scoreboard 5.55 (mbarrier
try_wait spins on Q/dO arrivals — TMA/L2 latency), wait 1.67,
everything else <1.2. Next levers:
- dQ smem mailbox drained by producer warps 1-3 (FA4 does this —
  barrier ids dQEmptyWG0/dQFullWG0 with 128+32 threads) to take
  even the red.v2 drain off the MMA path.
- tile_m=80 like FA4 (25% fewer iterations; needs n80 wgmma and a
  different sdS layout — their dS smem is k-major, trans-b=0).
- deeper Q/dO ring (8 slots) if smem allows, to absorb TMA latency.
- preprocess (551us vs FA4 153) and convert (835 vs 109) are still
  3-7x off; both are pure-bandwidth kernels worth one coalescing
  pass (currently ~0.4ms combined of the 4ms gap).

## Not yet tried (fwd)

- L2 cache hints on TMA loads (FA4 emits
  `cp.async.bulk.tensor...L2::cache_hint`; our stdlib path emits
  none — would need stdlib patch or inline PTX).
- 32-bit (`layout_int_type=DType.int32`) LayoutTensor index types
  (PTX still has ~50 add.s64 vs FA4's 0).
- Profile-guided softmax reordering with SASS (`--set source`).
- Locking clocks for stable A/B (`sudo nvidia-smi -lgc <freq>`).

## Legacy

The FA2-targeting `fwd/`, `fwd_fa3/`, `bwd/` subpackages and their
tests/benches predate the FA4 race and are throwaway-grade reference
(user call). Clean up once FA4 parity is declared.
