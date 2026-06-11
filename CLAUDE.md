# CLAUDE.md

Guidance for working on this repo: FlashAttention-4-class attention
kernels written from scratch in Mojo.

## State (2026-06-11): parity with FlashAttention-4 on H100

The package races Tri Dao's **FlashAttention-4** (`flash_attn.cute`,
CuTe DSL). Original minimalist config — bf16, head_dim=128,
non-causal, contiguous, seqlen % 128 == 0, Hq == Hk; the envelope
has since grown (causal, GQA, varlen+ragged, fp16 — see below). Canonical benchmark shape:
**B=2, S=8192, H=16, D=128**. Both kernels are AT PARITY within
run-to-run variance (locked clocks, interleaved):

- fwd: 0.991–0.992x (mojo FASTER; 2226 µs vs FA4 2202 after the
  2026-06-11 O-store epilogue fix — see below)
- bwd: 6148–6250 µs vs FA4 5913–6176 (1.00–1.06x); preprocess and
  dq-convert at parity too.
- GQA (2026-06-10, Hq % Hkv == 0, fully differentiable; canonical
  Hq=16/Hkv=4): fwd 1.034x plain / 0.984x causal; bwd 0.980x plain
  / 1.005x causal. fwd = h_kv = h_q // ratio on the K/V TMA coords
  (comptime ratio variant). bwd = FA4's fp32-accum design: grid
  stays per-q-head; the epilogue stages dK/dV as row-major f32 in
  the dead K+V smem (exactly 64 KiB) and cp.reduce.async.bulk-adds
  into per-kv-head accumulators (cross-CTA L2 reduction, no
  atomics); preprocess zeroes them; a torch permute-cast converts.
- CAUSAL (2026-06-10, both differentiable end-to-end): fwd 0.986x
  post-epilogue-fix (the LPT scheduler was the original parity
  gate); bwd 3146–3242 vs 3129–3204 (0.988–1.048x across 6 runs —
  straddles 1.0). Causal bwd uses FA4's tile_m=64 but KEEPS our
  swapped dQ^T (deliberate: same wgmma inventory, identical mailbox;
  FA4's unswapped form + per-wg column split is documented in the
  causal-bwd audit if ever needed). Causal bwd scheduler: plain 3-D
  grid — FA4 uses NO LPT for the bwd (PTX-verified).
- VARLEN (2026-06-11, packed cu_seqlens, `flash_attn_varlen_func`,
  fully differentiable): fwd AND bwd at parity at the canonical
  mixed 16384-token config (fwd 0.997–1.001x / 0.978–0.981x causal;
  bwd 0.981x / 1.003x causal — mojo faster on most). The varlen
  machinery itself is FREE (degenerate-uniform [8192,8192] = 0.988x,
  mojo faster). Design: host work-item
  tables (int32[8] rows; fwd/preprocess/convert per q-tile, bwd
  main per kv-tile) whose addresses ride existing kernel arg slots
  (sched_swizzle in the fwd, dk_accum_ptr in the bwd) so kernel
  signatures stay byte-identical to dense (gated per edit); FA4's
  padded-packed stats layout (`padded_offset_q =
  ((cu_q[i]+i*BM)//BM)*BM`, host-precomputed — no device //80);
  packed (H, total_q) LSE; our +inf-LSE/0-dpsum per-seq window
  padding (NOT FA4's lse=0 + in-loop mask — the two are coupled;
  documented at the preprocess write site). ARBITRARY LENGTHS
  (2026-06-11, every config at parity: aligned + ragged mixed all
  ~1.00x, causal fwd 0.98x): ragged kv tails are masked
  boundary-tile-only — the fwd walks kv tiles in REVERSE under
  varlen non-causal so the boundary tile lands in the consumer
  PROLOGUE and the steady loop stays mask-free (an in-loop mask
  branch cost 3.5%; online softmax is order-independent); causal
  fwd needs NO kv mask (the diagonal mask subsumes it — BM == BN +
  self-attn); the bwd masks S^T kv rows every m-trip on boundary
  CTAs (causality does NOT subsume there). Partial tail tiles store
  c-frags straight to gmem row-predicated (raw ptrs: O rides
  sched_num_hb_q; dk/dv ride an aux work-table row at index
  grid_dim.x). Envelope: any lengths >= 1, self-attn lengths, MHA or
  GQA, fully differentiable. All per-CTA table scalars are
  warp.broadcast-laundered. VARLEN GQA (2026-06-11): fwd at parity
  (0.988-0.989x, mojo faster); bwd trails 1.04x plain / 1.07x causal
  — the ONLY config above wobble. Root cause (per-kernel CUPTI
  decomposition): the per-CTA GQA epilogue fixed cost (64 KiB smem
  staging + 2 serialized cp.reduce, ~3 us/CTA) amortizes poorly over
  varlen's short per-CTA m-sweeps, + accumulator zeroing (+48 us in
  preprocess) + the torch permute-cast (+43 us); FA4 restructures
  varlen-GQA (pack_gqa / CLC re-enabled — the audit noted it). Known
  levers if parity is later required: pack-GQA-style M-dim head
  packing, or a fused dkv-convert kernel. Design note: varlen GQA
  needs NO per-seq padded accumulator windows (deliberate divergence
  from FA4's padded_offset_k) — masked S^T garbage rows give
  EXACTLY-ZERO c-frags, so full-tile cp.reduce adds into neighbour
  rows are numeric no-ops; only the buffer END pads to a full tile.
  Accum ptrs + total_k_alloc ride the aux table rows; the epilogue
  REBUILDS its addresses from the table/special regs at issue time
  (keeping them live across the main loop spilled 8-16 B).

- SHORT-SEQ FWD (2026-06-11, resolved): the fwd trailed 1.05–1.22x
  at short sequences (B=1 S=512 single-wave: +2.5 µs/CTA,
  PC-sampling-attributed ~8% to 16 serialized UTMASTG.3D issues +
  UTMACMDFLUSH from the unswizzled 16B-chunk O descriptor, plus the
  EXIT-drain it caused). Fix: stmatrix-stage O into the dead
  (swizzled) Q tile and issue ONE SWIZZLE_128B whole-tile TMA store
  — the bwd dK/dV epilogue's proven pattern. This also made the
  CANONICAL fwd faster than FA4 (0.991x) and closed the varlen
  mixed-config gap. Residual: B=32 S=512 (4-trip, 36-wave) is still
  ~1.05x — known, low value. The lesson generalizes: descriptor
  chunking shows up as serialized UTMASTG issue cost, ~70 cycles
  apiece, exposed whenever per-CTA work is small.

- FP16 (2026-06-11): the kernels were dtype-parameterized all
  along; two stdlib over-restrictions had to be vendored around —
  `st_matrix` comptime-asserts bf16/f32 (stmatrix.b16 is
  dtype-agnostic: pointer-bitcast at the call sites) and the
  register-A (RS) `wgmma_async` overload is hardcoded `.bf16.bf16`
  inline asm (`_wgmma_f16.mojo` vendors the m64n128k16 f32.f16.f16
  arm; the three RS sites — fwd PV, bwd dV/dK — fork under
  `comptime dtype == float16`, replicating the TensorCoreAsync RS
  k-loop; SS sites go through the dtype-generic NVVM intrinsic).
  bf16 codegen byte-identical; fp16 parity fwd 1.01-1.03x / bwd
  1.016x (in-band); 0 spills.

`HANDOFF.md` is the full race log: architecture, the perf journey,
the codegen lessons (uniform-register file capacity, the
tid-widening trap, descriptor rematerialization), the measurement
protocol, and the complete negative-results list — **read it before
attempting any perf change**.

## Measurement protocol (non-negotiable)

1. `sudo nvidia-smi --lock-gpu-clocks=1500,1500` first (persistence
   mode is off — relock every session). Unlocked, this H100 drifts
   ±4–5% and fakes wins/losses below ~3%.
2. Even locked, both kernels wobble ~2–4% run-to-run: always bench
   A/B interleaved (`master_bench.sh` does) and quote 3-run spreads.
3. After every kernel edit:
   `ptxas -arch=sm_90a -v ptx/mojo_*.ptx` — the spill-bytes line is
   the canary (both kernels must stay at 0).

## Iteration tooling

- **Master script: `scripts/master_bench.sh [--kind bwd] [--causal]
  [--hkv N] [--varlen] [--quick] [--no-ncu]`**
  — clears the flash_attn_mojo JIT cache (not the mojo compiler
  cache), recompiles, runs correctness checks, benches mojo vs FA4
  interleaved (CUPTI kernel-only time), dumps the mojo kernel's PTX
  to `ptx/`, prints a PTX op-mix diff vs the committed FA4 reference
  and (unless `--no-ncu`) side-by-side ncu stats.
- **Fast correctness loop**:
  `uv run python scripts/bench_fa4.py --impl mojo --kind {fwd,bwd}
  --check-only [--causal] [--hkv N] [--varlen]` — delegates to
  pytest `tests/test_kernels.py -k "<kind> and <dense|varlen> and
  <plain|causal> and <mha|gqa>"` (the test/param names encode those
  axes, so you can also call pytest directly). The checks are
  fp32-reference comparisons via `flash_attn_mojo.reference`
  (`flash_attn_ref` / `flash_attn_varlen_ref`, both with
  `return_lse`): S ∈ {128, 256, 640, 1024} dense (640 = 8×80
  exercises the bwd tile_m=80 exact-fit path), tile-aligned varlen
  sets, plus canonical-bench-shape cross-checks vs `flash_attn.cute`.
  `FLASH_ATTN_MOJO_TEST_IMPL=fa4` runs the same reference checks
  against Tri Dao's kernels (harness validation). `--varlen-lens`
  sets the BENCH lengths (default: the canonical mixed 16384-token
  config).
- **`scripts/ptxas_ur_probe.py`** — generates toy wgmma-loop PTX in
  varying dataflow shapes, compiles with ptxas, reports
  UR-allocation / R2UR / spills. Use it to test any codegen
  hypothesis BEFORE touching the kernels (it found both the UR-file
  cliff and the tid-widening trap).
- **SASS op-mix diff** — extract both kernels' innermost loop bodies
  (same HGMMA count = same tensor work), diff opcode histograms.
  `IMAD.U32 R,RZ,RZ,URx` = UR→R move; `R2UR` storms mean a
  uniformity taint; see HANDOFF for the decoding table.
- **FA4 reference PTX**: `reference_ptx/` (committed; see its README
  for target numbers and regeneration). FA4's actual cubins can be
  dumped with `CUTE_DSL_KEEP=cubin`.
- **PTX dump plumbing**: `MOJO_DUMP_PTX=<path>` in the env; `_jit.py`
  forwards it as a `-D` define and `launch.mojo` passes it to
  `compile_function(dump_asm=...)`.

## Repository layout

- `src/flash_attn_mojo/`
  - `fwd_fa4/`, `bwd_fa4/`: the kernels (pure JIT-on-first-use; no
    AOT sweep). Each subpackage:
    - `kernel.mojo` — device function(s), comptime-parameterized.
    - `common.mojo` — shared constants (tile sizes, stage counts).
    - `launch.mojo` — TMA descriptor + smem setup,
      `compile_function` + `enqueue_function`.
    - `variant.mojo` — static entry point reading comptime params
      via `std.sys.get_defined_*`; exports `PyInit_variant`.
    - `_jit.py` — config extraction → `-D` defines → shared
      cache+compile+load helper.
    - `__init__.py` — Python wrapper building the args tuple.
    The bwd is FA4's 3-kernel pipeline: preprocess (dpsum, lse·log2e,
    dq_accum zeroing) → main (dK/dV + dQ mailbox/cp.reduce drain) →
    convert (dq_accum fragment-dump decode → bf16 dq).
  - `_jit_common.py`: shared variant cache + compile + load helper;
    owns the env-signature → cache-hash logic.
  - `_fn.py`: the public `flash_attn_func` autograd op (+ packed
    wrappers). Envelope-checked; non-CUDA tensors fall through to
    the reference.
  - `reference.py`: pure-PyTorch `flash_attn_ref` (SDPA-based).
- `tests/`: `uv run pytest tests/` — `test_api.py` API/envelope
  errors + CPU reference path; `test_fa4.py` public-API (autograd)
  correctness vs the fp32 references; `test_kernels.py`
  kernel-wrapper checks per variant (what the bench delegates to,
  incl. canonical-shape cross-checks vs `flash_attn.cute`). All
  reference math lives in `reference.py` — never re-implement it
  inline.
- `compat/`: drop-in `import flash_attn` shim package
  (`flash-attn-mojo-compatibility`).
- `flash-attention/`: gitignored clone of Tri Dao's repo (the
  `flash_attn/cute/` CuTe DSL source is the algorithm reference:
  `flash_fwd_sm90.py`, `flash_bwd_sm90.py`).
- `./modular`: the modular repo pinned at `d86df2b645`
  (`mojo/v1.0.0b1` == `max/v26.3.0` tags) — matches the pinned
  wheels, so stdlib/kernel sources there are exactly what we compile
  against. Key paths: `mojo/stdlib/std/gpu/`,
  `max/kernels/src/layout/{tma_async,tensor_core_async}.mojo`.

## Toolchain notes

- Mojo compiles PTX→cubin in-process via the statically linked
  `modular/lib/libNVPTX.so` (ptxas 13.1.115). Override with
  `MODULAR_NVPTX_COMPILER_PATH` (our `__init__.py` auto-points it at
  the `nvidia-cuda-nvcc-cu12` wheel if installed). FA4's cute DSL
  embeds ptxas 12.9.83; the two produce byte-identical SASS on our
  PTX — toolchain version is NOT a perf variable here.
- `uv run --extra nvidia` is currently broken in this venv (the
  upstream flash-attn 2 wheel is cp312, venv is cp313). FA4 itself
  (`flash_attn.cute`) imports fine in the plain venv via the local
  clone + `nvidia-cutlass-dsl`.

## Profiling

`scripts/profile_kernel.sh` wraps Nsight Compute (`ncu`) so you can
trace a single kernel launch without fighting JIT or warmup
pollution (bracketed capture via `cudaProfilerStart/Stop`,
`--profile-from-start no`). Defaults assume an H100 with
`RmProfilingAdminOnly=1` but passwordless sudo (the wrapper
auto-elevates).

For stall attribution use PC sampling:
`ncu --section SourceCounters -o rep ...` then
`ncu --import rep --page source --print-source sass --csv`, sort by
'Warp Stall Sampling (All Samples)'. Sector tables
(`l1tex__t_sectors_..._op_st` per request) catch coalescing bugs.

Section sets while iterating: `--set basic` (SOL + scheduler +
launch, ~10 passes), `--set detailed` (+memory/compute/occupancy),
`--set full` (everything, ~30–40 passes).

## Cache invalidation

You should never need to clear the cache manually — the env
signature (Python ABI, mojo version, modular SDK path, CPU brand,
ptxas signature) auto-invalidates on env shifts. If you suspect
something stale anyway:

```bash
rm -rf ~/.cache/flash_attn_mojo/
```

See `_jit_common.py::_env_signature` for the authoritative key list.
Production pattern: pre-warm the cache on a staging host, bundle it,
set `FLASH_ATTN_MOJO_USE_CACHE_ONLY` to turn cache misses into loud
errors instead of silent JIT compiles.

## Mojo/Hopper gotchas (hard-won — don't relearn)

Codegen (the ones that cost the most time; full stories in
HANDOFF.md and the memory notes):

- **Uniform-register file capacity**: the UR file is 63/warp; ≥~16
  simultaneously-live 64-bit descriptors overflow it, ptxas spills
  the overflow to LOCAL and reloads+R2URs per HGMMA. LLVM loves
  hoisting loop-invariant descriptor variants — launder the smem
  roots through a no-op `mov.b32` inline asm inside the loop so
  descriptors are REBUILT per iteration (rematerialization is free
  on the uniform datapath; liveness kills).
- **The tid-widening trap**: `Int(thread_idx.x) // 128` widens tid
  before the shift; ptxas's tid-uniformity rule only matches 32-bit
  `shr.u32`, and LLVM re-canonicalizes 32-bit extracts back to
  64-bit. Use `warp.broadcast(Int32(tid >> 7))` for any
  warp/warpgroup index (convergent: LLVM keeps it; recognized
  broadcast: ptxas uniformizes).
- Mojo `(x - y) // 2` on Int is a SIGNED floor-div: a 17-op rounding
  correction chain per smem address. Use `(x >> 1) - (y >> 1)`.
- The smem 128B-swizzle XOR is computed from the ABSOLUTE address:
  `addr ^ ((addr >> 3) & 112)`. For strided stmatrix sequences
  re-apply it per call unless the step provably avoids addr bits
  4–6 (2048-B steps are safe, 32-B steps are not).
- `stack_allocation` scalar arrays in kernels land in LOCAL memory
  (LDL/STL = long_scoreboard stalls) even with comptime indices —
  read from smem at use sites or use SIMD values.
- `SIMD[f32, 40]` (any non-power-of-2 width) is a comptime assert —
  use the `StaticTuple` `wgmma_async` overload for n=80 c-regs.
- LLVM CSEs adjacent identical `wgmma.wait_group` intrinsics and
  floats mbarrier arrives across wgmma inline asm — source-order
  wgmma scheduling is not guaranteed.
- ptxas honors `setmaxnreg`: pool = 2·128·240 + 128·24 = 384·168.
  Don't trust ncu's "registers/thread" (the static 168) as the
  consumer budget.
- The stdlib `cp_async_bulk_reduce_global_shared_cta` is wrongly
  comptime-gated to SM100+ — keep the inline asm (constraints
  "l,r,r"). Plain 1-D `cp.async.bulk` G2S with mbarrier has no
  stdlib wrapper either ("r,l,r,r").
- `dump_asm` paths must be `StaticString(...)`-wrapped.
- `DType` has no `.size_of()`; use `std.sys.size_of[dtype]()`.
- TMA OOB rows zero-fill on loads (the stdlib hardcodes
  `OOB_FILL_NONE`); our 3-D descriptors flatten (B·S) so only the
  last batch's tail truly zero-fills — interior batch tails read the
  next batch's rows, which the +inf-LSE / 0-dpsum padding
  annihilates exactly (finite-garbage assumption).

## Extending the envelope (if/when)

The natural next features, in rough order of value: other head
dims, the varlen-GQA bwd gap (pack-GQA-style head packing — see the
State note), dense arbitrary seqlen routing through the varlen
path. The FA4-class algorithm core
(warp specialization, tile_m=80 bwd, the mailbox dQ drain) carries
over. Keep every change inside the measurement protocol above — and
add new seqlen/shape cases to `bench_fa4.py --check-only` and
`tests/test_fa4.py`, not ad-hoc scripts.
