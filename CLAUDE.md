# CLAUDE.md

Guidance for working on this repo: FlashAttention-4-class attention
kernels written from scratch in Mojo.

## State (2026-06-10): parity with FlashAttention-4 on H100

The package races Tri Dao's **FlashAttention-4** (`flash_attn.cute`,
CuTe DSL) at one minimalist config — bf16, head_dim=128, non-causal,
contiguous, seqlen % 128 == 0, Hq == Hk. Canonical benchmark shape:
**B=2, S=8192, H=16, D=128**. Both kernels are AT PARITY within
run-to-run variance (locked clocks, interleaved):

- fwd: 2237–2276 µs vs FA4 2206–2255 (1.00–1.03x)
- bwd: 6148–6250 µs vs FA4 5913–6176 (1.00–1.06x); preprocess and
  dq-convert at parity too.

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

- **Master script: `scripts/master_bench.sh [--kind bwd] [--no-ncu]`**
  — clears the flash_attn_mojo JIT cache (not the mojo compiler
  cache), recompiles, runs correctness checks, benches mojo vs FA4
  interleaved (CUPTI kernel-only time), dumps the mojo kernel's PTX
  to `ptx/`, prints a PTX op-mix diff vs the committed FA4 reference
  and (unless `--no-ncu`) side-by-side ncu stats.
- **Fast correctness loop**:
  `uv run python scripts/bench_fa4.py --impl mojo --kind {fwd,bwd}
  --check-only` — fp32-reference checks at S ∈ {128, 256, 640, 1024}
  (640 = 8×80 exercises the bwd tile_m=80 exact-fit path; the others
  leave partial tail m-tiles). fwd also checks LSE.
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
- `tests/`: `uv run pytest tests/` — API/envelope tests (CPU) +
  CUDA correctness vs fp32 reference + cross-check vs
  `flash_attn.cute` when importable.
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

The natural next features, in rough order of value: causal masking
(block-skip in both kernels + masked softmax), other head dims,
MQA/GQA (Hk < Hq indexing on the K/V TMA coords), varlen. The
FA4-class algorithm core (warp specialization, tile_m=80 bwd, the
mailbox dQ drain) carries over; these are predicate/indexing
variations. Keep every change inside the measurement protocol above
— and add new seqlen/shape cases to `bench_fa4.py --check-only` and
`tests/test_fa4.py`, not ad-hoc scripts.
