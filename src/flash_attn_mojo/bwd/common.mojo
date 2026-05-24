"""Shared comptime constants for the bwd subpackage.

The main bwd kernel uses a per-(kv_block, batch, head) grid and walks
the Q axis in BM-sized tiles. Tile sizes match the fwd kernel.

MVP shape: BM=64, BN=64, head_dim=64 (locked at variant compile time).
"""


# 128 threads per block — 4 warps × 32 lanes. Matches the fwd kernel and
# Tri Dao's bwd block size for head_dim<=64.
comptime kBwdNThreads: Int = 128
comptime kBwdBlockM: Int = 64
comptime kBwdBlockN: Int = 64
