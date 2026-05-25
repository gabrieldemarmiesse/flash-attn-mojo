# flash_attn_mojo — handoff notes

Snapshot of where this port stands, what's left, and what changes
when you have a Hopper / H100 to work on.

Branch: `init_flash_attn_mojo` (35 commits ahead of `main`).
Run tests: `uv run --extra nvidia pytest`.
Run bench: `uv run --extra nvidia python benchmarks/bench_gpu_kernel_time.py`.

## Current correctness state

Both the fwd and the bwd are **functionally complete** within the
documented envelope. 147 of 148 tests pass; the 1 failure
(`test_basic.py::test_cuda_outside_envelope_rejects_clearly`) is
stale — it expects fp16 to raise NotImplementedError, but fp16 now
works via the API-boundary cast. Update the assertion or delete the
test.

## Current perf state (RTX 2000 Ada, sm89)

**Forward** is at perf parity with upstream flash-attn 2:

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

### 1. bwd perf rewrite — the biggest item

Convert the bwd's 5 matmuls from scalar loops to `multistage_mma`
+ tensor cores. A prior agent attempt converting just one matmul
(S=Q·K^T) regressed perf because the swizzled-smem cost wasn't
amortized; **all 5 matmuls must convert together** to share the
swizzled buffers.

Key pieces (`src/flash_attn_mojo/bwd/kernel.mojo`):
- Q smem: row_major(BM, D), swizzled. Used by S=Q·K^T and dK+=dS^T·Q.
- K smem: row_major(BN, D), swizzled. Used by S=Q·K^T and dQ+=dS·K.
- V smem: row_major(BN, D), swizzled. Used by dP=dO·V^T.
- dO smem: row_major(BM, D), swizzled. Used by dV+=P^T·dO and dP=dO·V^T.
- P smem: row_major(BM, BN), bf16 cast from fp32 S after softmax/mask.
  Consumed by dV (as A operand) and dQ (as part of dS).
- dS smem: row_major(BM, BN), bf16 cast from fp32 dS. Consumed by dK.

Smem budget at hd=64: Q+K+V+dO = 32 KiB, P+dS = 16 KiB → 48 KiB.
Plus pipeline staging adds another ~16 KiB → ~64 KiB total. Fits Ada
and H100. At hd=128: doubles to ~96 KiB on Ada (right at the cap);
H100 has 228 KiB available so it fits easily.

Reference: study `src/flash_attn_mojo/fwd/kernel.mojo`'s
`multistage_mma` invocations — same template params, same swizzle
patterns. The bwd has 5 matmuls instead of 2 but the per-matmul
pattern is identical.

Expected speedup: from ~89-384x upstream down to ~1-2x upstream
(based on fwd parity). 100-300x perf gain.

This is M-H session work. Plan it as one big commit (all 5 matmuls
together) OR a sequence: one commit per matmul where each commit
keeps the rest scalar but the smem layout is already set up for the
full conversion.

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
