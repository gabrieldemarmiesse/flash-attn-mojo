"""Shared constants for the Apple-GPU (Metal) forward attention kernel.

The blocking mirrors ccv/metal-flash-attention on apple9 (M1–M4):
parallelization=16 q rows per threadgroup, traversal=128 kv rows per
step, 2 simdgroups (64 threads) each owning 8 q rows. See
METAL_PLAN.md ("v1 fight — resolved") for the derivation and the
three perf mechanisms.
"""

comptime BR = 16  # q rows per threadgroup (parallelization dimension)
comptime BC = 128  # kv rows per traversal block
comptime TPB = 64  # 2 simdgroups; each owns 8 q rows
comptime LOG2E = 1.4426950408889634

# Residency throttle: an (otherwise unused) threadgroup allocation of
# this many f16 elements (~10.7 KiB) caps resident threadgroups at
# 3/core. The kernel uses no functional threadgroup memory, so
# unthrottled residency over-subscribes and desynced K/V readers blow
# the SLC once per-head K+V is large. Worth ~36% at S=8192 D=128.
comptime RESIDENCY_PAD_ELEMS = 5460
