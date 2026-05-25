# flash_attn_mojo — handoff notes

Snapshot of where this port stands, what's left, and what to attack next.

Branch: `main` (the prior `init_flash_attn_mojo` branch was merged).
Run tests: `uv run --extra nvidia pytest`.
Run bench: `uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py`.

## Resuming the bwd perf work (last touched 2026-05-25)

**Recent commits on `main` (newest first):**
```
ee4ca06 bwd: pad PT/dST smem rows by 8 bf16 — 7-8% perf win
6f11490 docs: CLAUDE.md — document the profiling workflow
1658850 scripts: ncu profiling helpers (profile_kernel.sh + bench harness)
d95cc49 deps: pin flash-attn cp312 wheel (was cp313)
51486a7 docs: HANDOFF.md — capture session pause + ncu unblock plan
64ce67e bwd: deterministic dqaccum (no atomic-add) — 11-15x perf win
b9794c3 bwd: tensor-core MMA rewrite (multistage_mma for all 5 matmuls)
```

Combined effect: bwd at `(1,1024,8,64)` is **171 μs** (was 140 μs before
the padding fix per the prior HANDOFF table, but that measurement appears
to have been on a different bench run — the post-vectorize-stores baseline
was 186 μs, padding takes it to 171 μs, a clean -8%). Upstream FA3 on
the same shape is 50 μs, so we're 3.38x off. See current perf table below.

**State at last session pause:** all 134 functional tests pass (1 stale
test for fp16-NotImplemented is unrelated). The bank-conflict warning
that previously dominated the c-frag → smem stores (4-way on 75% of
shared stores, ncu est. 23.84% local speedup) is now absent from the
ncu report. Two failed/dropped experiments from earlier sessions:

- **Single-buffer Q/dO/K layout** (one (M, D) row-major buffer per
  tensor, both A-chunked and B-striped iter views over it): broke
  correctness (max-abs diff 3.3 vs reference). Suspected swizzle
  mismatch between async-copy writes and `multistage_mma`'s load_a
  reads when the warp-tile sub-tile origin shifts. Reverted in the
  same session. The win would be ~24 KiB less smem + half the per-qb
  gmem load traffic — small impact (gmem traffic isn't the bottleneck
  per the diagnostic), so this is low priority.
- **BK=64 for D≥64**: no measurable change vs BK=32, reverted.

### Profiling

The profiling toolchain is now committed — see `CLAUDE.md` "Profiling"
section and `scripts/README.md`. Quick recipe:

```bash
# capture: ncu wrapper auto-elevates if RmProfilingAdminOnly=1
scripts/profile_kernel.sh --kernel bwd-main -- --kind bwd --shape 1,1024,8,64
# summarize:
scripts/profile_summary.sh /tmp/kernel_bwd_kernel_prof.ncu-rep
```

### Top opportunities from the current ncu report (post-padding)

Captured on H100 PCIe at `(B=1, L=1024, H=8, D=64)`, bwd main kernel
duration 267 μs. ncu "Est. local speedup" numbers do not compound
across rows — they're independent rough estimates.

| signal | local speedup | what's actually going on |
|---|---|---|
| Theoretical occupancy 12.5% (gates scheduler) | 75% | 255 regs/thread + 76 KiB smem/block both limit to 2 blocks/SM. Need to drop one to ~170 regs *and* shave ~7 KiB smem to reach 3 blocks/SM. |
| Fixed-latency stalls 34% of issue cycles | 34% | back-to-back register-dependent fp32 (softmax/dropout/alibi inner loops). Restructure for ILP or lean on occupancy fix. |
| Local mem loads/stores at 1/32 byte sectors | 20-22% | register spilling under the 255-cap. Compiler spilled state to local. |
| Uncoalesced global stores (47% sector util) | 10-20% | dQaccum and dK/dV gmem writes are vectorized as SIMD[T,2] but still touch only 16 of 32 bytes per sector. The c-frag → gmem pattern fundamentally doesn't coalesce a full warp; this is mostly cosmetic vs the bigger items. |
| Workload imbalance | 14% | Some SMs see ~17% more cycles than average. Probably grid/block scheduling under low waves-per-SM (0.56). Helped by occupancy. |
| Uncoalesced shared loads (20% excess wavefronts) | 6-10% | multistage_mma's load_a access pattern. Was 45% before padding; cut in half. Probably not worth chasing standalone. |

### Concrete next steps

The dominant remaining bottleneck is **register pressure → low
occupancy**. None of the easy wins ship in a single commit; pick one
of the following structural attacks for the next session:

1. **Force fewer registers via `maxnreg`.** PTX/SASS supports
   `.maxnreg N` per-function to cap per-thread registers, trading
   forced spills for more blocks/SM. Mojo exposes
   `@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=...)` but I
   could not find a `maxnreg` equivalent in the stdlib. Worth
   asking Modular or hand-writing the LLVM intrinsic. If we can
   cap at ~170 regs/thread, occupancy doubles → potentially close
   most of the gap to upstream.

2. **Eliminate `dq_contrib` register tile** (32 fp32 regs/thread).
   It lives only between MMA 4 and the gmem write of dqaccum and
   could conceptually be done in-place over an already-dead tile
   (s_reg after the dST write, dp_reg after the dS combine). The
   blocker is that multistage_mma's c-frag layout is determined by
   the call params, so aliasing requires the layouts to match. Same
   `c_frag_size=4` for all 5 MMAs but the warp-tile shape differs.
   Probably needs a custom inner MMA loop for the dQ step.

3. **Restructure to 8 warps / 256 threads** like Tri Dao FA2 does at
   hdim=64. Halves per-thread c-frag size, splitting BN in half
   between warp pairs. Larger structural change: every warp index
   calculation flips, the smem buffers may need re-sizing, and 256
   threads * 255 regs = 65,280 regs > 64K SM budget → 1 block/SM
   (worse than current 2!). Only useful if combined with item 1
   (a `maxnreg` cap).

4. **FA3 / WGMMA / TMA port for Hopper.** Multi-week effort. Would
   be the path to <1x upstream, but the FA2-pattern multistage_mma
   path can probably still be pushed to ~1.5-2x off upstream with
   items 1-3.

One-shot iteration loop (clears JIT cache, runs tests, benches):

```bash
rm -rf ~/.cache/flash_attn_mojo/bwd \
  && uv run --extra nvidia pytest tests/ -q \
       --deselect tests/test_basic.py::test_cuda_outside_envelope_rejects_clearly \
  && uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py \
       --iters 100 --warmup 5 --mode bwd
```


## Current correctness state

Both the fwd and the bwd are **functionally complete** within the
documented envelope. 147 of 148 tests pass; the 1 failure
(`test_basic.py::test_cuda_outside_envelope_rejects_clearly`) is
stale — it expects fp16 to raise NotImplementedError, but fp16 now
works via the API-boundary cast. Update the assertion or delete the
test.

## Current perf state (H100, sm90)

Captured 2026-05-25 with `bench_gpu_kernel_time.py --iters 20`. Upstream
on H100 dispatches the FA3 (sm90) kernel for fwd, so the fwd ratio is
no longer at parity — that's the FA3-port work item (item 3 below).

**Forward** vs upstream FA3 (H100):

```
shape (B,L,H,D)   | mojo us | upstream us | ratio
(1, 128, 8, 64)   |    8.76 |        5.25 | 1.67x
(1, 512, 8, 64)   |   22.81 |       10.05 | 2.27x
(1, 1024, 8, 64)  |   41.44 |       16.66 | 2.49x
(1, 2048, 8, 64)  |   81.86 |       37.48 | 2.18x
(2, 1024, 8, 64)  |   46.00 |       20.87 | 2.20x
(4, 1024, 8, 64)  |   87.22 |       31.77 | 2.75x
(8, 1024, 8, 64)  |  193.04 |       61.10 | 3.16x
(1, 4096, 8, 64)  |  308.10 |      110.14 | 2.80x
(1, 8192, 8, 64)  | 1419.29 |      421.97 | 3.36x
```

**Backward** (after the PT/dST padding fix, 2026-05-25):

```
shape (B,L,H,D)   | mojo bwd | upstream FA3 | ratio
(1, 128, 8, 64)   |    22 us |       15 us  |  1.43x
(1, 512, 8, 64)   |    73 us |       30 us  |  2.42x
(1, 1024, 8, 64)  |   171 us |       51 us  |  3.38x
(1, 2048, 8, 64)  |   592 us |      182 us  |  3.25x
(2, 1024, 8, 64)  |   308 us |      100 us  |  3.08x
(4, 1024, 8, 64)  |   478 us |      155 us  |  3.08x
```

Improvements since the last HANDOFF snapshot:
- **PT/dST smem padding** (this session): 7-8% across all shapes.
  Per-row padding of 8 bf16 elements breaks the BK-bank-aligned
  write pattern that previously produced 4-way bank conflicts on
  75% of c-frag → smem stores. The ncu shared-store-conflict
  warning is now absent from the report; the shared-load
  warning is also cut in half (45% → 20% excess wavefronts).
- **Vectorized gmem stores** (this session): SIMD[T,2] packs each
  pair of c-frag elements into a single `st.global.b64` (dQaccum)
  or `.b32` (dK/dV). No measurable perf change — confirms gmem
  stores were never the bottleneck — but removes one ncu warning
  and one source of instruction-count overhead.

Headline gain from the two bwd rewrites:
- **MMA rewrite** (commit `b9794c3`): 3.7-5.5x over scalar (replaced
  5 scalar matmul inner loops with `multistage_mma` tensor-core calls).
- **Deterministic dqaccum**: another **11-15x** on top (this commit).
  Diagnostic-traced finding: under the prior atomic-add design, each
  (q_row, d) cell of dqaccum was atomic-added concurrently by every
  n_block in the grid (16-way contention at L=1024). Replacing atomic
  add with a per-n_block slot in an expanded dqaccum (shape
  (num_n_blocks, B, H, L, D), zeroed via torch.zeros) plus a sum-then-
  cast in the convert_dq kernel completely removes that contention.
  Verified by replacing the atomic-add with a no-op: 1024-shape kernel
  time dropped from 2119 μs → 116 μs, validating that the atomics were
  ~95% of kernel time. The proper implementation lands at 140 μs (an
  extra ~24 μs vs the no-op upper bound: write traffic to the expanded
  dqaccum + the convert_dq reduction).

Combined: bwd is now **40-65x faster than the scalar baseline** and
**1.4-3.8x off upstream FA3 on H100** (down from 59-250x).

Likely sources of the remaining 1.4-3.4x gap, in rough impact order:

1. **Register pressure → low occupancy.** Mojo codegen hits the
   255-regs-per-thread cap; the spill manifests as local-mem
   traffic at 1/32 byte sector utilization. Limits to 2 blocks/SM
   (7.1% achieved occupancy). See "Concrete next steps" above.

2. **Fixed-latency execution stalls.** 34% of issue cycles. Tight
   serial fp32 chains in the softmax/dropout/alibi combine loops.
   Mostly tied to (1) — more concurrent warps would hide these.

3. **Redundant per-qb gmem loads.** Q, dO, K each get loaded twice
   from gmem (chunked-A view + striped-B view). The second load
   hits L2 not HBM, so cost is small (~5 μs at L=1024) but non-zero.

4. **FA3 (Hopper-only) opportunity.** Upstream uses FA3 (TMA +
   WGMMA + warp specialization) on H100; we still use FA2-pattern
   multistage_mma. Multi-week port; would be the path to <1x
   upstream.

5. **dqaccum scaling**: At large seqlen the expanded dqaccum memory grows
   linearly with `num_n_blocks`. For (1, 8192, 8, 64), 128 n_blocks ×
   2 MB = 256 MB — fine for typical workloads but a concern at extreme
   shapes. Tri Dao caps the deterministic factor (typically at 128 or
   `min(num_n_blocks, K)`); we currently use full N. A bounded variant
   would trade some atomic contention back for memory.

## Prior perf state (RTX 2000 Ada, sm89, pre-H100)

**Forward** was at perf parity with upstream flash-attn 2:

```
shape (B,L,H,D)   | mojo us | upstream us | ratio
(1, 128, 8, 64)   |    8.01 |        7.90 | 1.01x
(1, 512, 8, 64)   |   32.37 |       33.52 | 0.97x  ← faster
(1, 1024, 8, 64)  |  104.73 |       91.99 | 1.14x
(1, 2048, 8, 64)  |  355.04 |      343.37 | 1.03x
(2, 1024, 8, 64)  |  146.10 |      140.90 | 1.04x
(4, 1024, 8, 64)  |  293.73 |      318.38 | 0.92x  ← faster
(8, 1024, 8, 64)  |  789.47 |      647.81 | 1.22x
(1, 4096, 8, 64)  | 1337.30 |     1251.79 | 1.07x
(1, 8192, 8, 64)  | 5736.46 |     4737.68 | 1.21x
```

Geomean 1.06x, faster at 2 of 9 shapes.

**Backward** is correct but ~89–384x slower than upstream:

```
shape (B,L,H,D)   | mojo bwd us | upstream bwd us | ratio
(1, 128, 8, 64)   |      1753   |              20 |   89x
(1, 512, 8, 64)   |     18977   |              63 |  302x
(1, 1024, 8, 64)  |     64203   |             167 |  384x
(1, 2048, 8, 64)  |    198204   |             681 |  291x
(2, 1024, 8, 64)  |    122702   |             331 |  371x
(4, 1024, 8, 64)  |    232015   |             829 |  280x
```

Why: the bwd kernel does its 5 matmuls (S=Q·K^T, dV+=P^T·dO, dP=dO·V^T,
dK+=dS^T·Q, dQ+=dS·K) as **scalar fp32 arithmetic with smem-staged
intermediates** rather than tensor-core MMAs. Correctness first, perf
second — the perf rewrite is the headline remaining work item.

## Feature coverage (vs upstream flash-attn 2)

| feature | fwd | bwd |
|---|---|---|
| bf16 native | ✓ | ✓ |
| fp16 (API cast to bf16, returns fp16) | ✓ | ✓ |
| head_dim ∈ {32, 64, 128} native | ✓ | ✓ |
| head_dim round-up dispatch (any mult-of-8, ≤128) | ✓ | ✓ (autograd inherits) |
| head_dim ∈ {96, 160, 192, 224, 256} | **blocked** | **blocked** |
| causal (with block-skip) | ✓ | ✓ |
| MQA/GQA | ✓ | ✓ |
| softcap | ✓ | ✓ |
| sliding window (with block-skip) | ✓ | ✓ |
| ALiBi (per-head and per-batch-head) | ✓ | ✓ |
| dropout (deterministic seed/offset) | ✓ | ✓ (RNG replay) |
| return_attn_probs (LSE) | ✓ | n/a |
| qkvpacked / kvpacked wrappers | ✓ | autograd inherits |
| flash_attn_varlen_func (prefill-only) | ✓ | autograd inherits |
| flash_attn_with_kvcache (prefill-only) | ✓ | n/a |
| non-contig (L, D) strides on unaligned seqlens | ✓ | ✓ |

## Remaining work

### 1. bwd perf rewrite — DONE (commit `b9794c3`)

All 5 bwd matmuls converted from scalar fp32 inner loops to
`multistage_mma` tensor-core MMAs. Smem layout uses 9 buffers (each
of Q, K, dO gets an A-chunked view AND a B-striped view because they
play both transpose_b roles across the 5 matmuls; PT and dST get
dedicated transposed-layout buffers written from m16n8k16 C-fragments
after softmax / dS combine; V is single-layout). dQ MMA uses A-from-
registers (same trick fwd uses for P·V), atomic-adding c-fragments to
dqaccum.

Smem at hd=64: ~72.5 KiB; hd=128: ~144.5 KiB (fits H100, exceeds Ada
99 KiB cap — would need a smaller-block path for Ada D=128, not yet
implemented since H100 is the current target).

Delivered speedup: 3.7-5.5x over scalar; brought from ~60-250x to
~16-56x upstream. The remaining gap to upstream is documented under
"Current perf state" above. Next-step optimization ideas there are
estimates only — none have been profiled with ncu (no permission for
GPU perf counters on this box; only nsys-level timing available).

### 2. head_dim ∈ {96, 160, 192, 224, 256}

Blocked on the V smem swizzle constraint: `Layout.row_major(BK, depth)`
makes V's row stride = depth, and `multistage_mma`'s `load_b` requires
the row stride be a power of 2 in {16, 32, 64, 128, 256, 512}
half-elements. Depths 96/160/192/224 fail. Depth 256 fits the
swizzle but blows the Ada 99 KiB dynamic-smem cap.

On H100 this changes:
- **Smem cap is 228 KiB** (vs Ada's 99 KiB) — depth 256 with BN=64
  fits easily. Just add to the allowlist and test.
- **Swizzle constraint is the same** — depths 96/160/192/224 still
  blocked at compile time inside multistage_mma. Unlocking these
  needs either:
  - Restructure V smem into multiple BN-wide depth strips, splitting
    the P·V MMA into per-strip iterations. Real algorithmic change.
  - Hand-roll the second MMA inner loop without multistage_mma (use
    direct mma.sync intrinsics).
  - Use FA3's WGMMA path (Hopper only) which uses a different smem
    layout that may sidestep the constraint.

### 3. FA3 port (Hopper-only)

This is the headline H100 opportunity. FA3 (Colfax + Tri Dao, 2024)
rewrites the FA2 algorithm around three Hopper-only hardware features:

1. **TMA** (Tensor Memory Accelerator) for async bulk loads via
   tensor-descriptor addressing. Replaces `cp.async`.
2. **WGMMA** (warpgroup matrix multiply) — 4-warp warpgroup matmuls
   with async issue and larger fragment sizes.
3. **Warp specialization** — producer warpgroup (does loads) +
   consumer warpgroup (does compute) overlap in the same CTA.

Expected speedup on H100: 1.5-2.0x over FA2 fwd. ~75% of theoretical
fp16 peak vs FA2's ~35%.

Mojo stdlib exposes the necessary primitives:
- `tma.bulk_async_*` for TMA loads (look in
  `modular/mojo/stdlib/std/gpu/...` for the exact module).
- `wgmma_async` in `linalg/matmul/gpu/` for WGMMA.
- Hopper-specific barriers (`mbarrier_arrive_expect_tx`, `mbarrier_wait_parity`)
  for the producer-consumer handshake.

Approach:
1. Start a new subpackage `src/flash_attn_mojo/fwd_fa3/` (parallel to
   `fwd/`) so the FA2 path stays as the fallback for non-Hopper.
2. Port from Tri Dao's `flash-attention/hopper/` (separate codebase
   from the sm80 kernels in `csrc/flash_attn/src/`).
3. Dispatch in `_fn.py`: when running on sm90+, prefer FA3.

Reference vendored sources (already cloned):
- `flash-attention/hopper/flash_fwd_kernel_sm90.h` — FA3 fwd
- `flash-attention/hopper/flash_bwd_kernel_sm90.h` — FA3 bwd
- MAX itself uses FA3 for sm90 head_dims {64, 72, 80, 96, 128, 256, 512}.
  See `modular/max/kernels/src/nn/attention/gpu/mha.mojo` —
  `_is_sm10x_gpu(info)` / `info == H100` gates select the FA3 path.
  This is the most relevant reference for how to wire it up in Mojo.

This is multi-week work, not a single session.

### 4. Other items, smaller

- **dropout on the pytorch fallback bwd**: currently raises
  NotImplementedError because faithfully replaying upstream's dropout
  RNG in pytorch is nontrivial. The native bwd handles dropout via
  RNG replay from the saved (seed, offset). If we ever need the
  pytorch fallback to support dropout (debugging), this is the work.
- **flash_attn_with_kvcache decode mode** (`seqlen_q == 1,
  seqlen_kv >> 1`): currently raises NotImplementedError because the
  kernel requires `seqlen_q == seqlen_k`. Needs a separate kernel
  variant that allows different Q and K seqlens. Used in autoregressive
  inference. Should also support `rotary_cos/sin`, `cache_batch_idx`,
  `cache_leftpad`, `block_table` once that's in place.
- **fp16 native path** (vs the current cast-to-bf16 wrapper) is
  blocked on the mojo-compiler 1.0.0b1 stdlib not shipping the
  m16n8k16 fp16 intrinsic. `get_mma_shape[fp16, fp32]` returns
  m16n8k16, and only the bf16 variant exists at our pinned mojo
  version. A mojo bump (or a hand-rolled m16n8k8 fp16 path) unblocks
  this. H100 has the same intrinsic availability — this is purely a
  toolchain blocker, not hardware.
- **Update the stale fp16-rejection test** in
  `tests/test_basic.py::test_cuda_outside_envelope_rejects_clearly`.

## Suggested H100 work order

1. **Quick wins** (1-2 commits):
   - Move dev to the H100, re-run the test suite, fix any platform-
     specific issues (`ptxas` version, smem cap detection, etc.).
   - Add head_dim 192 and 256 to the allowlist (smem cap no longer
     blocks them on H100). Validate against upstream.
   - Update the stale fp16-rejection test.

2. **bwd perf rewrite** (M-H session): convert the 5 bwd matmuls to
   `multistage_mma` together. Should get the bwd from ~300x upstream
   to ~1-2x upstream on Ada (the FA2 ceiling). On H100 the result is
   the FA2 baseline before any FA3 work.

3. **FA3 port** (multi-week): start with the fwd, then the bwd. New
   subpackage; dispatch in `_fn.py` based on GPU compute capability.

4. **head_dim {96, 160, 224}**: revisit once FA3 is in place — the
   WGMMA path may not have the same swizzle constraint as
   `multistage_mma`, in which case these unlock automatically.

## Repo orientation

```
src/flash_attn_mojo/
├── _fn.py               # public API + autograd Function
├── reference.py         # pure-PyTorch SDPA fallback for CPU
├── fwd/
│   ├── kernel.mojo      # the fwd device function (FA2 multistage_mma)
│   ├── launch.mojo      # host launcher (DeviceContext + compile + enqueue)
│   ├── variant.mojo     # static entry point, -D-define-driven
│   ├── common.mojo      # comptime tile constants
│   ├── _jit.py          # python JIT shim
│   └── __init__.py      # native_fwd entry point
├── bwd/
│   ├── kernel.mojo      # bwd main kernel (scalar matmuls, perf-pending)
│   ├── preprocess.mojo  # delta + dQaccum-clear
│   ├── convert_dq.mojo  # fp32 dQaccum → bf16/fp16 dq
│   ├── launch.mojo, *_launch.mojo, *_variant.mojo, _*_jit.py
│   └── __init__.py
└── _jit_common.py       # shared variant cache + compile + load
tests/
├── test_fwd.py          # 88 fwd tests
├── test_bwd.py          # 42 bwd tests
└── test_basic.py        # smoke tests (1 stale failure to fix)
benchmarks/
└── bench_gpu_kernel_time.py
compat/
└── src/flash_attn/      # drop-in import shim
flash-attention/         # vendored Tri Dao CUDA reference (gitignored)
├── csrc/flash_attn/src/ # FA2 sm80 kernels — primary reference
└── hopper/              # FA3 sm90 kernels — for the H100 work
modular/                 # vendored modular/MAX source at tag max/v26.3.0
└── max/kernels/src/nn/attention/gpu/mha.mojo   # MHAConfig / FA3 dispatch
```

## Quick sanity checklist before continuing

- `git status` clean; on branch `init_flash_attn_mojo`.
- `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`
  shows your H100 / sm90.
- `uv run --extra nvidia pytest tests/test_fwd.py tests/test_bwd.py`
  shows 130 passed (88 fwd + 42 bwd).
- `uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py`
  runs (numbers will differ vs Ada — H100 is much faster).
- Clear the JIT cache if anything looks off:
  `rm -rf ~/.cache/flash_attn_mojo/`.

## fwd_fa3 — Hopper FA3 fwd port (in progress)

**Recent commits on `main` (newest first):**
```
fa1e496 fwd_fa3: cleaner output store loop; isolate bug to per-row variation
4364577 fwd_fa3: revert PV transpose_b experiment (broke shape interpretation)
db69e94 fwd_fa3: online softmax via lane_group reductions
d21980e fwd_fa3: TMA + WGMMA pipeline runs end-to-end (correctness pending)
da986a3 fwd_fa3: scaffold (sm_90+ Hopper subpackage, kernel is a no-op stub)
```

**Status:** scaffold + TMA + WGMMA × 2 + online softmax + gmem write are all
landed and the kernel compiles and runs on H100. Output is wrong by a
per-row factor:

```
mojo[i, 0, 0, 0] for i=0..3: [1.78, 1.92, 1.70, 2.17]
ref[i, 0, 0, 0]  for i=0..3: [1.00, 1.00, 1.00, 1.00]   (V = ones probe)
```

Mean across all rows = 1.0 (correct on average) but per-row variance is real.
Not a bf16-precision issue (upstream FA3 matches fp32 to 2e-3). Not a
swizzle issue (SWIZZLE_NONE produces bit-identical output to SWIZZLE_128B).

**Diagnosis:** the c-frag scalar layout was verified via modular's
`p_vec_output_layout` (mha.mojo:1086): scalar order per thread is
`(top, top, bot, bot)` per col-chunk, repeating across 16 col-chunks
for m64n128, totaling 64 elements. My current row mapping
`m_mma*2 + (1 if (c % 4) >= 2 else 0)` matches this. (A mid-session
experiment using `(c // NC) & 1` was wrong and has been reverted.)

Bug is therefore NOT in c-frag indexing. Further probes narrowed it:

```
Q=K=zero,   V=ones:  output = 1.0 exactly (perfect)
Q=rand, K=zero, V=ones: 1.0 exactly      (S = 0, identical to above)
Q=0, K=rand, V=ones:    1.0 exactly      (S = 0)
Q=K=rand, V=ones:       1.78, 1.92, 1.70, 2.17 (broken; per-row variance)
```

**Decisive probe (skipped normalization, V=ones, Q=K=rand):** mojo
output per row was 2.19, 2.09, 2.16, 2.19 (constant across cols
within each row, std=0 across the 64 cols). Python-computed true
unnormalized rowsum for the same Q is ~1.0-1.3 — so the wgmma_pv
is producing **~2× the expected rowsum**. The factor is close to
2 but not exactly constant across rows.

This rules out:
- a per-element layout bug (would not produce constant-across-cols)
- a missing reduction (would produce row-varying, but probably
  not a ~2× factor)

This suggests:
- a wgmma accumulation that double-counts somehow (running the
  inner k-loop twice, or accumulating m_mma=0 and m_mma=1 into
  the wrong output slot), OR
- the c-frag of QK being applied to the PV wgmma in a way that
  duplicates parts of P along the K axis.

So the kernel is correct whenever S is uniformly 0. Per-row variation
in S → per-row error in O. This rules out static structure issues
(c-frag walk, c→a reshape, output store, warp/lane mapping all
behave identically whether S is constant or not). What changes between
the two cases is the *per-row state flow* through the softmax:
rowmax / rowsum / scale_old correction. At L=128 there is only 1
KV block so scale_old can't be the issue — leaving:

1. The `warp.lane_group_max[num_lanes=4]` / `lane_group_sum[num_lanes=4]`
   reductions. These should broadcast to all 4 participating lanes, but
   maybe not in the configuration we use. Worth verifying by computing
   a known reduction (e.g., set local_max[0] = lane_id and check the
   post-reduce value on each lane equals max of lane_ids in the group).

2. The wgmma_pv accumulation under bf16-cast P with varying magnitudes.
   When P values span many orders of magnitude (because S values do),
   the bf16 representation loses precision for small values and the
   wgmma sum may behave differently per row. But the magnitude of the
   error (~0.75) is far larger than expected bf16 noise.

3. Most plausible: an interaction between the wgmma_pv's K-direction
   slicing of P (8 k_iters of 16 cols each) and the c-frag → a-frag
   "view" — even though the byte order matches, maybe the WGMMA
   expects the A operand to have specific REGISTER FILE positions
   (not just memory positions) for the wgmma.descriptor calc. The
   LayoutTensor's stack_allocation might not give registers in the
   contiguous order WGMMA expects across k_iters.

**Concrete next steps for the next session:**

1. **Device-side print probe.** Set s_reg to a known per-(row, col)
   pattern (e.g. `s_reg.ptr[i] = lane * 100 + i`) immediately *after*
   the wgmma_qk, then read it back and confirm the stored values
   correspond to the (row, col) positions you expect. This unambiguously
   verifies the c-frag layout assumption.

2. **Replace `p_reg.copy_from(s_reg.reshape(...))` with an explicit
   per-element walk** that computes (row, col) for each c-frag index
   and writes to the corresponding a-frag slot. If this fixes
   correctness, the issue is element ordering; if not, look at the
   wgmma_pv descriptor.

3. **Sanity-check by setting P = constant 1/128 (skip the actual
   softmax)** — then with V=ones, O should equal 1.0 regardless of
   the c-frag/a-frag mapping (since all P values are identical). If
   this gives 1.0, the bug is definitively in the per-row softmax
   path; if not, the bug is in the wgmma_pv → output flow.

**Key files:**
- `src/flash_attn_mojo/fwd_fa3/kernel.mojo` — softmax c-frag walk at
  lines 293-322 (rowmax) and 339-352 (rowsum); output store at
  lines ~390-440.
- `src/flash_attn_mojo/fwd_fa3/launch.mojo` — TMA descriptor setup
  and kernel launch.

**Debug recipe:**
```bash
rm -rf ~/.cache/flash_attn_mojo/fwd_fa3
uv run --extra nvidia python -c "
import torch
from flash_attn_mojo.fwd_fa3 import native_fwd_fa3
from flash_attn import flash_attn_func
B,L,H,D = 1,128,1,64
torch.manual_seed(0)
q = torch.randn(B,L,H,D,dtype=torch.bfloat16,device='cuda').contiguous()
k = q.clone(); v = torch.ones_like(q)
out = torch.zeros_like(q); native_fwd_fa3(q,k,v,out,softmax_scale=1.0/(D**0.5))
ref = flash_attn_func(q,k,v)
print('mojo[:4,0,0]:', out[0,:4,0,0].tolist())
print('ref [:4,0,0]:', ref[0,:4,0,0].tolist())
"
# Target: both show all 1.0
```
