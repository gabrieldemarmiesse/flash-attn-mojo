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

## Backward (bwd_fa4) — state as of 2026-06-10

Correct (bf16 noise floor vs fp32 autograd, 2.4e-4 vs FA4 grads) at
**24.3 ms vs FA4's 6.5 ms (3.77x)**. Loop: `scripts/master_bench.sh
--kind bwd`; fast check: `scripts/bench_fa4.py --impl mojo --kind
bwd --check-only`.

Perf journey: 141.8 ms -> 28.5 ms (acquire->relaxed atomics; the
default `Atomic.fetch_add` ordering emits `atom.acquire` — always
pass `ordering=RELAXED`, and value-returning `atom` is still slower
than `red`; the kernel now uses inline-asm
`red.relaxed.gpu.global.add.v2.f32`) -> 24.3 ms (manual sdS
addressing instead of LayoutTensor setitem crd2idx, coalesced
preprocess/convert, smem-staged lse/dpsum prefetched 1 tile ahead,
deferred wgmma waits, 1 barrier/iter).

Found races worth remembering: (1) the epilogue stages dK/dV into
the K/V smem areas — wg1 can reach it while wg0's *last* dQ GEMM
still reads kt_view -> pre-epilogue named barrier required. (2) sdS
double-buffering alone is not enough without it.

Probes: disabling the whole dQ path (GEMM+wait+atomics) only saves
~4 ms -> the core S^T/dP^T/softmax/dV/dK loop is itself ~3x too
slow. Tensor pipe is ~16% busy (FA4: ~53%). The fix is FA4's bwd
overlap schedule (commit next tile's S^T before processing the
current softmax, à la fwd v3->v4) + balancing dQ across both
warpgroups:

- **dQ^T = K^T·dS split over 2 warpgroups needs an mn-major A
  operand. modular's TensorCoreAsync only does k-major A, but the
  raw `std.gpu.compute.mma.wgmma_async` exposes `layout_a="col"` —
  hand-roll the dQ^T GEMM with `_wgmma_descriptor` built the way
  TensorCoreAsync builds its transpose_b=False B descriptor.** Then
  dS^T is stored in its NATURAL c-frag orientation (paired stores,
  no transpose scatter), dq_accum becomes (B,H,D,S) so dQ^T c-frag
  pairs stay contiguous for red.v2, and convert does a smem-tile
  transpose.
- FA4 uses tile_m=80 (m64n80 wgmma) precisely to fit the register
  budget when everything overlaps; revisit BM after the schedule
  rework.

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
