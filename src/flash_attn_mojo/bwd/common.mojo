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
# BK chunk size for all 5 bwd MMAs. Multiple of MMA_K (=16 for bf16
# m16n8k16) and a divisor of BM, BN, head_dim. Kept here so launch.mojo
# can size the PT/dST smem correctly with PT padding.
comptime kBwdBlockK: Int = 32
# Extra bf16 elements per (BN, BK) row in the PT/dST chunks. Breaks the
# BK-bank-aligned write pattern that otherwise produces 4-way bank
# conflicts on c-frag stores (75% of shared stores per ncu). 8 keeps the
# row 16-byte aligned for ldmatrix while shifting bank residues.
comptime kBwdPtPad: Int = 8
