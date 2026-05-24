"""Shared comptime constants for the fwd subpackage.

Tile sizes are picked to match modular's `mha_single_batch` config
for non-MQA bf16 on Ampere/Ada (num_warps_m=4, num_warps_n=1).
At head_dim=64:
    BM = 64   queries per block
    BN = 64   keys per inner KV tile (= depth)
    BK = 32   reduction (head_dim) tile per multistage_mma step
    WM = 16   queries per warp (= MMA_M)
    WN = 64   keys per warp     (= BN, since num_warps_n == 1)
At head_dim=128: BN/WN bumped to 128 (BN == depth, so the second MMA
P·V's output (BM, depth) fits cleanly into one warp tile (WM, WN==BN)
without splitting the depth axis across warps). Smem grows to ~96 KiB,
which requires the Ada/Ampere dynamic-smem opt-in
(MAX_DYNAMIC_SHARED_SIZE_BYTES) — handled by `launch_fwd`.
"""


comptime kNThreads: Int = 128  # 4 warps × 32 = num_warps_m * num_warps_n * WARP_SIZE

comptime kBlockM: Int = 64
comptime kBlockK: Int = 32

comptime kWM: Int = 16


# Per-head-dim BN/WN: BN == WN == depth so the second MMA's output tile
# (BM, depth) fits one warp's WN cols (num_warps_n == 1). Defined in
# kernel.mojo and launch.mojo as `BN = head_dim`.
