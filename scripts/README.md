# scripts/ — profiling helpers

Helpers for capturing ncu (Nsight Compute) traces of the flash-attn-mojo
kernels. The fwd is at perf parity already; the bwd is the active perf
target — see `HANDOFF.md` for the per-shape gap vs upstream FA3.

## The one-command perf gate: `master_bench.py`

`scripts/master_bench.py` is the high-information, autonomous perf gate —
run it after any kernel edit and read the summary. One invocation locks
the GPU clocks (hard gate), clears the JIT cache + runs correctness,
benches mojo vs FA4 with a per-shape ratio table (ratio | run-to-run
spread | verdict) and a compute (tensor-core) roofline (achieved TFLOP/s,
% of peak, regime + hint), captures an ncu metrics summary for both
kernels side by side, dumps the mojo PTX, diffs its instruction mix vs
the committed FA4 reference, runs the ptxas spill canary (hard gate), and
does an independent wall-clock run. It ends with a machine-readable
`===AGENT-SUMMARY===` JSON block and a non-zero exit on any gate failure.

```bash
scripts/master_bench.py                   # canonical dense fwd (flash_attn_func)
scripts/master_bench.py --kind bwd        # backward kernels
scripts/master_bench.py --causal --hkv 4  # causal GQA; also --varlen/--window/--softcap
scripts/master_bench.py --full            # multi-shape sweep
scripts/master_bench.py --no-ncu --no-asm # timing-only fast loop
scripts/master_bench.py --no-lock --no-clean  # dev loop (keep JIT cache)
```

The older `master_bench.sh` is the original bash coordinator (fwd/bwd PTX
diff + ncu side-by-side); `master_bench.py` supersedes it with the richer
tables, roofline, spread verdicts, and agent summary.

## Quick start

```bash
# bwd main kernel, default shape (B=1, L=1024, H=8, D=64), full section set
scripts/profile_kernel.sh --kernel bwd-main -- --kind bwd

# fwd causal, custom shape
scripts/profile_kernel.sh --kernel fwd -- --kind fwd --shape 1,2048,8,64 --causal

# summarize a captured report
scripts/profile_summary.sh /tmp/kernel_bwd_kernel_prof.ncu-rep
```

## Files

- `profile_bench.py` — minimal one-shot bench harness. Runs fwd or bwd at
  a single (B, L, H, D) shape with warmup, then brackets the capture
  iters in `cudaProfilerStart` / `cudaProfilerStop` so ncu's
  `--profile-from-start no` excludes JIT and warmup launches.
- `profile_kernel.sh` — ncu wrapper around `profile_bench.py`. Handles
  the `RmProfilingAdminOnly` driver gate (auto-elevates via sudo when
  the gate is locked and passwordless sudo is available), resolves
  pixi/uv to absolute paths so they survive `sudo -E`, and picks
  sensible defaults for `--set`, `--kernel-name`, and the output path.
- `profile_summary.sh` — thin `ncu --import` wrapper for already-captured
  `.ncu-rep` files. Defaults to the `details` page.

## Kernel filter shortcuts

`--kernel` resolves to an ncu `--kernel-name regex:...` pattern. Pick by
intent rather than by mangled name:

| `--kernel`        | regex                          | what it captures              |
|-------------------|--------------------------------|-------------------------------|
| `fwd`             | `kernel_fwd_kernel`            | mojo fwd                      |
| `bwd`             | `bwd`                          | mojo bwd (all 3 sub-kernels)  |
| `bwd-main`        | `kernel_bwd_kernel`            | mojo bwd main (dK/dV/dQaccum) |
| `bwd-preprocess`  | `preprocess_bwd_preprocess`    | mojo bwd preprocess           |
| `bwd-convert`     | `convert_dq_bwd_convert_dq`    | mojo bwd convert_dq           |

To profile something else (e.g. upstream's `flash_fwd_kernel`), pass
`--filter '<your regex>'` directly. If neither is set, the wrapper
derives the filter from `--kind`.

## The RmProfilingAdminOnly gate

On most cloud GPU hosts, `/proc/driver/nvidia/params` ships with
`RmProfilingAdminOnly: 1`, which makes ncu's metric-based sections
(Memory, Occupancy, SchedulerStats, ...) return `ERR_NVGPUCTRPERM` for
non-root users. Two ways to unlock:

```bash
# per-reboot (lasts until the next nvidia.ko reload):
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

# or — far simpler — run ncu under sudo. profile_kernel.sh does this
# automatically when passwordless sudo is configured.
```

Check the current state with:

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
# 0 → unlocked, 1 → need sudo / driver reload
```

## Section sets

`--set full` (default) pulls every section — ~6-30 MiB report, ~30
passes per kernel (so ~30x slower than a single kernel run). Pick a
smaller set when iterating on a hypothesis:

- `--set basic`     — SOL + SchedulerStats + LaunchStats. Tiny, fast.
- `--set detailed`  — adds Memory + Compute + Occupancy. Middle ground.
- `--set source`    — adds SASS/PTX-level metric overlay. Heavy.

## Opening reports

The CLI summary is fine for occupancy / SOL / bank-conflict checks:

```bash
scripts/profile_summary.sh /tmp/bwd_main_prof.ncu-rep
```

For source-level drill-down, copy the report off-box and open in
`ncu-ui` locally — the CLI can't render the source view.

## ROCm / AMD (MI300X) lane — `master_bench_rocm.sh`

The AMD analog of `master_bench.sh`, for the CDNA (gfx942) race. There is
no FlashAttention-4 / CuTe on AMD, so the reference baseline is
**CK-flash** — Tri Dao's `flash_attn` built with the **Composable Kernel**
backend (the default ROCm backend of the flash-attention repo). It is the
**only** reference: if `flash_attn` is not installed the harness errors
out (build it per the setup below).

CK was chosen because it is the fastest attention kernel on this MI300X.
Against the two other AMD options — PyTorch fused SDPA (ROCm's AOTriton
flash backend) and the AMD Triton FA2 kernels (the `aiter` package's
`flash_attn_triton_amd`, the flash-attention README's Triton backend) —
the measured kernel-only ranking is **CK > Triton > SDPA** everywhere:

| shape (fwd, fp16)        | SDPA        | Triton      | CK-flash    | CK vs SDPA |
|--------------------------|-------------|-------------|-------------|------------|
| 4096 × 16 × 128          | 238 TFLOP/s | 310 TFLOP/s | 363 TFLOP/s | 1.52×      |
| 4096 × 16 × 128 causal   | 138         | 188         | 246         | 1.78×      |
| 8192 × 16 × 128          | 243         | 333         | 388         | 1.60×      |
| 2048 × 16 × 64           | 105         | 124         | 173         | 1.64×      |

(So CK is the baseline; SDPA/Triton are not benched by the harness. The
one-off comparison that produced this table was run by forcing each
backend; the Triton kernels were vendored standalone as `fa_triton_amd`,
which is no longer required by the harness.)

The mojo lane is **v0** —
`bench/bench_mojo_rocm.mojo`, a hand-vectorized SIMD forward kernel
(wavefront-64, `BK=64`) that does **not** use the CDNA matrix cores
(MFMA) yet, so it is far behind the reference. This is the M0 milestone
(correctness + measurement plumbing), the analog of the Metal race's M0.

One invocation mirrors the five master-bench steps:

1. recompile the mojo v0 kernel from source;
2. run mojo v0 (fp32 CPU correctness + wall-clock time) and the CK
   reference (kernel-only via roctracer, the CUPTI analog);
3. copy the mojo kernel's **AMDGCN ISA** dump (the PTX analog, written by
   `dump_asm` on every run) into `asm/`;
4. print the GCN instruction-mix + resource footprint (`gcn_opmix.py`) —
   the `vgpr_spill_count` is the spill canary (analog of the ptxas
   spill-bytes line), and a `matrix` class of 0 confirms v0 uses no MFMA;
5. re-time the mojo kernel under `rocprofv3` for a kernel-only number +
   launch resources (`rocprof_summary.py`, the ncu-stats analog).

```bash
scripts/master_bench_rocm.sh                 # default 4096 x 16 x D128
scripts/master_bench_rocm.sh --quick         # small shape, fast
scripts/master_bench_rocm.sh --head-dim 64   # D=64 variant
scripts/master_bench_rocm.sh --no-prof       # skip the rocprofv3 step
```

Supporting scripts:

- `bench_rocm.py` — CK reference lane; benches the CK `flash_attn` forward
  kernel-only via `torch.profiler` (roctracer), emits a `RESULT` line in
  the same format as `bench_fa4.py`. Exits if `flash_attn` is not installed.
- `gcn_opmix.py` — AMDGCN ISA op-mix histogram / diff (analog of
  `ptx_stats.py` / `air_opmix.py`): classes opcodes (`v_mfma`→matrix,
  `ds_`→lds, `ds_bpermute`→shuffle, `global_`→gmem, `scratch_`→spill,
  `s_waitcnt`→sync) and parses the vgpr/sgpr/LDS/scratch footprint.
- `rocprof_summary.py` — groups a `rocprofv3 --kernel-trace` CSV by kernel
  name and prints mean/min device time + VGPR/scratch/LDS/grid.

### Environment setup (once, on the AMD box)

The default `uv sync` installs the CUDA toolchain (the `dev` group's
`flash-attn-4[cu13]`). For the ROCm lane install a ROCm PyTorch into the
venv instead — verified working on ROCm 7.2.4 / MI300X:

```bash
uv pip install "torch==2.8.0" --index-url \
    https://download.pytorch.org/whl/rocm6.4 --reinstall-package torch
```

Mojo itself targets gfx942 out of the box on this toolchain
(`DeviceContext().api()` reports `hip`).

Then build the **CK-flash** reference (the default ROCm backend of the
flash-attention repo — verified on ROCm 7.2.4 / MI300X, ~17 min). The
clone lives at `flash-attention/` (gitignored). `GPU_ARCHS`/`OPT_DIM`
restrict the (large) instantiation set to what the bench uses:

```bash
git clone https://github.com/dao-ailab/flash-attention   # if not present
cd flash-attention
git submodule update --init --depth 1 csrc/composable_kernel
uv pip install ninja packaging setuptools wheel
GPU_ARCHS=gfx942 OPT_DIM=64,128 MAX_JOBS=20 ROCM_HOME=/opt/rocm \
  PATH=/opt/rocm/bin:$PATH VIRTUAL_ENV=$(pwd)/../.venv \
  uv pip install --no-build-isolation .
```

Leave `FLASH_ATTENTION_TRITON_AMD_ENABLE` unset to get the CK backend
(setting it `TRUE` selects the Triton/aiter backend instead). This CK
`flash_attn` is required — the harness errors out without it.

<details>
<summary>Reproducing the SDPA/Triton comparison (not needed by the harness)</summary>

The comparison table above was a one-off. SDPA needs nothing extra. For
the Triton FA2 kernels, `import aiter` triggers a heavy native JIT build,
but the kernel files are pure torch+triton with package-relative imports,
so vendor just that subpackage as a standalone top-level package:

```bash
git submodule update --init --depth 1 third_party/aiter   # in the clone
cp -r flash-attention/third_party/aiter/aiter/ops/triton/_triton_kernels/flash_attn_triton_amd \
      .venv/lib/python3.12/site-packages/fa_triton_amd
```

then call `fa_triton_amd`'s `interface_v2.fwd(...)` directly (see the
git history of `bench_rocm.py` for the exact call). The current harness
does not reference `fa_triton_amd`.
</details>
