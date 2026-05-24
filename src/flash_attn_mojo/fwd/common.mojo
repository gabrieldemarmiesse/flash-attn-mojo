"""Shared comptime constants for the fwd subpackage.

Tile sizes are picked to match modular's `mha_single_batch` config
for non-MQA bf16 on Ampere/Ada (num_warps_m=4, num_warps_n=1).

Per-head-dim BN (keys-axis tile width):
    head_dim ∈ {32, 64, 128} → BN = head_dim (bit-identical to the
        previous BN==depth code path).
    head_dim ∈ {192, 256}    → BN = 64. Decoupling BN from depth here
        keeps the K-smem tile (BN × depth) small enough to fit Ada's
        99 KiB dynamic-smem cap (Q smem 64×depth*2 + K smem 64×depth*2 +
        V smem BN×depth*2 + P smem BM×BN*2). At depth=256 the total is
        ~76 KiB.

WN follows the per-MMA N-axis:
    Q · Kᵀ:  WN = BN     (num_warps_n_keys  == 1, P stays in registers)
    P · V:   WN = depth  (num_warps_n_depth == 1, output stays in regs)
"""


comptime kNThreads: Int = 128  # 4 warps × 32 = num_warps_m * num_warps_n * WARP_SIZE

comptime kBlockM: Int = 64
comptime kBlockK: Int = 32

comptime kWM: Int = 16


@always_inline
fn kBlockN_for[head_dim: Int]() -> Int:
    """Pick the keys-axis tile width for a given head_dim."""
    @parameter
    if head_dim <= 128:
        return head_dim
    else:
        return 64
