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
