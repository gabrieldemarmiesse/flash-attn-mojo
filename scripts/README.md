# scripts/ — profiling helpers

Helpers for capturing ncu (Nsight Compute) traces of the flash-attn-mojo
kernels. The fwd is at perf parity already; the bwd is the active perf
target — see `HANDOFF.md` for the per-shape gap vs upstream FA3.

## The one-command perf gate: `master_bench.py`

`scripts/master_bench.py` is **the single, high-information, autonomous perf
gate — the one script an agent runs to see how to improve performance.** It
**auto-detects the GPU backend** and runs that backend's phases, ending with a
machine-readable `===AGENT-SUMMARY===` JSON block and a non-zero exit on any
gate failure. (It fuses the former per-backend coordinators — the CUDA
`master_bench.py`, the Apple `master_bench_metal.py`, and the ROCm
`master_bench_rocm.sh` — into one entry point.)

Backend selection (`--device {auto,cuda,rocm,metal,cpu}`, default `auto`):

| detected | how                                   | races                          |
|----------|---------------------------------------|--------------------------------|
| `cuda`   | `nvidia-smi` returns a device         | mojo vs FlashAttention-4       |
| `metal`  | `sys.platform == "darwin"`            | mojo vs best-of {MFA, ccv}     |
| `rocm`   | torch probe reports a HIP device      | mojo v0 vs CK-flash            |
| `cpu`    | `--device cpu` (or no GPU found)      | *no race* — reference smoke    |

Each backend's phases are detailed in the per-backend sections below. The
**CUDA** path is the richest (the full FA4 parity race + envelope: bwd, causal,
GQA, varlen, window, softcap, fp16, hdim64); **metal** and **rocm** are the v0
forward-only MMA fights; **cpu** has no kernel (the public op falls through to
the pure-PyTorch reference) so it is a correctness smoke-check plus a naive
reference wall-clock, *not* a perf race.

`master_bench.py` is uv-agnostic: it runs every child process with its own
interpreter (`sys.executable`), so you select the accelerator once, in the
**caller**, with a uv extra (torch is accelerator-specific and lives in the
mutually-exclusive `cpu`/`nvidia`/`rocm` extras — see `pyproject.toml`):

```bash
# H100 (CUDA): mojo vs FlashAttention-4
uv run --extra nvidia scripts/master_bench.py
uv run --extra nvidia scripts/master_bench.py --kind bwd          # backward kernels
uv run --extra nvidia scripts/master_bench.py --causal --hkv 4    # causal GQA; also --varlen/--window/--softcap
uv run --extra nvidia scripts/master_bench.py --full              # multi-shape sweep
uv run --extra nvidia scripts/master_bench.py --no-lock --no-clean  # dev loop (keep JIT cache)

# MI300X (ROCm): mojo v0 vs CK-flash. Use --no-sync so uv does not wipe the
# manually-built CK flash_attn (it is not an index-installable dependency).
uv run --extra rocm --no-sync scripts/master_bench.py --seq 4096 --head-dim 128

# CPU reference correctness + latency (no kernel race)
uv run --extra cpu scripts/master_bench.py --device cpu

# common flags: --no-prof / --no-asm (timing-only), --seq/--head-dim/--heads
```

(On macOS the metal backend needs no extra — darwin torch is a base
dependency. `uv run scripts/master_bench.py` just works there.)

The older `master_bench.sh` is retained only as the FA4 **reference-PTX
regeneration** utility (`--refresh-fa4-ptx`, see `reference_ptx/README.md`); it
is not a bench coordinator an agent needs — `master_bench.py` supersedes it for
all timing/correctness/profiling.

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

## Apple / Metal backend (`--device metal`, auto-detected on darwin)

The Metal forward-attention race. Unlike NVIDIA (one mojo kernel vs Tri Dao's
CUDA baseline) and unlike ccv's Apple path (mojo-only — "upstream is CUDA-only,
nothing to diff"), this machine has **two** hand-tuned references —
`philipturner/metal-flash-attention` (MFA) and ccv's C++ MFA port — so the
harness runs the full **3-way** comparison and takes the **per-shape best of
{MFA, ccv} as the baseline** (the empirical ceiling). ccv is the faster
reference at the shapes measured (~3.65-3.73 TFLOP/s vs MFA's ~2.6-2.9), so the
baseline is usually ccv.

`master_bench.py` runs these ccv-style phases on Metal and ends with a
machine-readable `===AGENT-SUMMARY===` JSON block:

- **(a) clock lock** — macOS has no clock-lock CLI, so this binary-patches a
  copy of Instruments' *Metal System Trace* template to force *Induced GPU
  Performance State = Maximum* (`_apple_gpu_clock_lock.py`). It applies to the
  (d) xctrace recording (verified: the profiler records at `Maximum` clock);
  the (c) bench self-times outside xctrace, so it is auxiliary there — drift is
  cancelled by interleaving instead. `--no-lock` opts out.
- **(b) correctness** — each lane's built-in fp32-reference `--check` at small
  shapes (hard gate; a broken softmax/mask shows at ≥1e-2).
- **(c) kernel bench** — interleaved 3-way; per-shape table of mojo / mfa / ccv
  µs, baseline = best(mfa,ccv), ratio, spread, achieved TFLOP/s, % of the
  ceiling, verdict + a compute-roofline hint. Attention is matmul-bound, so the
  best reference is the empirical ceiling. **The ratio is reported, not gated**
  (mojo is not expected to beat the MMA references during the v1 fight); the
  regression gate is a **ratchet** on mojo's own best per-shape time (under
  `--gate`, persisted in `scripts/baselines/` — gitignored, per-machine).
- **(d) profiler** (`--profile IMPL`) — one xctrace *Metal System Trace*
  recording at induced-max clock → per-encoder GPU intervals + Active/Idle
  **duty cycle** (a duty <40% flags launch/sync-bound; the v0 mojo kernel reads
  ~92% active = genuinely compute-bound). xctrace SIGSEGVs on finalize ~1 in 3;
  the harness retries and accepts the first parseable trace. HW counters
  (ALU%, occupancy, bandwidth) are GUI-only on Apple.
- **(e) introspection** — refreshes the mojo kernel's **AIR** dump and diffs its
  op-mix vs the committed MFA/ccv reference AIR (`air_opmix.py`; the Apple analog
  of the PTX histogram — `call air.simdgroup_matrix_8x8_*` = the HGMMA count;
  v0 has **zero** matrix ops, 128 `simd_shuffle` + 128 `convert` instead).

```bash
scripts/master_bench.py                       # quick tier (S∈{1024,4096}, D=128)
scripts/master_bench.py --full                # full sweep (adds S=8192 and D=64)
scripts/master_bench.py --seq 4096 --head-dim 128   # one shape
scripts/master_bench.py --gate                # fail on a mojo ratchet regression
scripts/master_bench.py --profile mojo        # + xctrace GPU intervals / duty
scripts/master_bench.py --no-build --no-asm   # fast dev loop
```

(On darwin these run automatically; pass `--device metal` to force it.)

Supporting scripts:

- `bench_metal.py` — the process-interleaved 3-way runner (mojo/MFA/ccv CLIs
  round-robin, pooled trials, min/median/spread/GFLOPS). Usable standalone for
  a single shape; `master_bench.py` calls its `bench_shape()` for the
  structured tables. See METAL_PLAN.md for the CLI contract and the timing fine
  print (mojo lane is wall-clock around enqueue+sync; refs are command-buffer
  GPU time — a conservative bias against mojo, <1% at canonical shapes).
- `_apple_gpu_clock_lock.py` — the induced-max template patcher (raw
  NSKeyedArchiver bplist patch; content-addressed cache under
  `~/.cache/flash_attn_mojo/`). Run standalone to print the template path.
- `xctrace_gpu_intervals.py` — robust `metal-gpu-intervals` parser (id/ref
  value-dictionary resolver, first-duration = GPU time, process filter, clock
  tagging, duty cycle). The scriptable subset of a Metal System Trace.
- `air_opmix.py` — AIR instruction-mix histogram / diff (the SASS/PTX op-mix
  analog); `xcrun metal-objdump -d` for .air, textual .ll passthrough.

## ROCm / AMD backend (`--device rocm`, auto-detected via a torch HIP probe)

The CDNA (gfx942) race. There is
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

`master_bench.py` mirrors the five master-bench steps on ROCm:

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
scripts/master_bench.py                      # auto-detects rocm; default 4096 x 16 x D128
scripts/master_bench.py --quick              # small shape, fast
scripts/master_bench.py --head-dim 64        # D=64 variant
scripts/master_bench.py --no-prof            # skip the rocprofv3 step
```

(Pass `--device rocm` to force it if the torch probe is ambiguous.)

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

torch is accelerator-specific, so it lives in mutually-exclusive uv extras
(`cpu`/`nvidia`/`rocm`, declared conflicting in `pyproject.toml`) — one
venv per accelerator. On the AMD box, sync the `rocm` extra (ROCm PyTorch
2.8.0 + `pytorch-triton-rocm`, from the rocm6.4 index):

```bash
uv sync --extra rocm      # torch==2.8.0+rocm6.4, pytorch-triton-rocm
```

Mojo itself targets gfx942 out of the box on this toolchain
(`DeviceContext().api()` reports `hip`).

Then build the **CK-flash** reference (the default ROCm backend of the
flash-attention repo — verified on ROCm 7.2.4 / MI300X, ~17 min). It is NOT
an index-installable dependency (it needs the composable_kernel submodule +
a hipcc build), so it is a manual step, not part of the `rocm` extra. The
clone lives at `flash-attention/` (gitignored); `GPU_ARCHS`/`OPT_DIM`
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

Because the CK build is manual (not tracked by uv), invoke the bench on
ROCm with **`--no-sync`** so uv does not wipe it while activating the extra:

```bash
uv run --extra rocm --no-sync scripts/master_bench.py --seq 4096 --head-dim 128
```

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
