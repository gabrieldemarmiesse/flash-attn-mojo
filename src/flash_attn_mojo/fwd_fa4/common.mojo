"""Shared comptime constants for the FA4-target fwd kernel.

Mirrors Tri Dao FA4's sm90 config for head_dim 128 (see
`reference_ptx/README.md`): tile 128x128, 2 MMA warpgroups + 1
producer warpgroup (TMA loads), 384 threads.
"""

comptime kFa4BlockM: Int = 128
comptime kFa4BlockN: Int = 128
comptime kFa4NMmaWarpgroups: Int = 2
comptime kFa4NThreads: Int = (kFa4NMmaWarpgroups + 1) * 128

# K/V shared-memory ring: K(n) lives in slot (2n) % kFa4KVStages,
# V(n) in slot (2n+1) % kFa4KVStages.
comptime kFa4KVStages: Int = 6
