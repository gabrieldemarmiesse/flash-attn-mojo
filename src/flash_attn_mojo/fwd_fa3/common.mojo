"""Shared comptime constants for the FA3 fwd subpackage.

FA3-specific block sizes:
- BM=128 (one WGMMA tile along M; matches Tri Dao FA3 at hdim=64)
- BN=128 (KV tile size; ample for pipelined TMA loads)
- 1 warpgroup = 128 threads (MVP, no producer/consumer split yet)

These will grow when warp specialization lands (typically 1 producer
warpgroup + 2 consumer warpgroups = 384 threads for FA3 fwd).
"""


# Single warpgroup MVP — 128 threads. Add producer wg (+128) once
# warp specialization lands.
comptime kFa3NThreads: Int = 128
comptime kFa3BlockM: Int = 128
comptime kFa3BlockN: Int = 128
