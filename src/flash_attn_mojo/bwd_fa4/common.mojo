"""Shared comptime constants for the FA4-target bwd kernels.

FA4's sm90 bwd config for head_dim 128 non-causal is tile_m=80,
tile_n=128 with SdP_swapAB + dQ_swapAB (see reference_ptx/README.md).
v1 uses tile_m=64 (keeps every wgmma shape m64nNk16 with
num_m_mmas=1 and lets the dQ GEMM run unswapped on one warpgroup).
"""

comptime kBwdBlockM: Int = 64  # Q rows per inner tile
comptime kBwdBlockN: Int = 128  # KV rows per block
comptime kBwdNMmaWarpgroups: Int = 2
comptime kBwdNThreads: Int = (kBwdNMmaWarpgroups + 1) * 128

# Q/dO shared-memory ring: Q(m) in slot (2m) % kBwdQdOStages, dO(m)
# in (2m+1) % kBwdQdOStages.
comptime kBwdQdOStages: Int = 6

# Preprocess / convert kernels: one Q-row per thread, 128 threads.
comptime kBwdPreBlockM: Int = 128
comptime kBwdPreThreads: Int = 128
