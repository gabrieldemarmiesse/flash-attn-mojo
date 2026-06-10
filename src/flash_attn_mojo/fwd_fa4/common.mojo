"""Shared comptime constants for the FA4-target fwd kernel.

Mirrors Tri Dao FA4's sm90 config for head_dim 128 (see
`reference_ptx/README.md`): tile 128x128, 2 MMA warpgroups.
v1 has no producer warpgroup yet — 256 threads total.
"""

comptime kFa4BlockM: Int = 128
comptime kFa4BlockN: Int = 128
comptime kFa4NMmaWarpgroups: Int = 2
comptime kFa4NThreads: Int = kFa4NMmaWarpgroups * 128
