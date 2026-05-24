"""Backward-preprocess kernel: delta = rowsum(dO * O), fp32, and
zero-fill the dQaccum workspace.

For each (b, h, q) compute
    delta[b, h, q] = sum_d dO[b, q, h, d] * O[b, q, h, d]

with the sum done in fp32 regardless of the input dtype. Output shape is
(B, H, L) row-major fp32 — matches the LSE tensor layout the bwd kernel
expects.

Additionally, the same block writes zeros to the dQaccum fp32
workspace at slice `[b, h, q_tile_row_base : q_tile_row_base + BM, :]`.
The main bwd kernel atomically accumulates dQ contributions into this
tensor across the KV blocks, so it must be cleared first. Mirrors Tri
Dao's flash_bwd_preprocess_kernel pipeline.

Tile config:
    BM         = 64 query rows per block
    num_threads= 128 (4 warps × 32 lanes)
    rows/warp  = BM / 4 = 16
    cols/lane  = head_dim / 32 (e.g. 2 at D=64, 4 at D=128, 1 at D=32)

Each warp loops over its 16 rows; for every row, all 32 lanes load
their cols/lane elements of dO and O (fp32-cast), multiply pairwise,
sum locally, then `warp.sum` reduces across the 32 lanes. Lane 0 of
each warp writes the result into `delta`.

For the dQaccum zero-fill, all 128 threads strip-mine the BM*D fp32
elements with stride 128 — each thread writes (BM*D)/128 = D/2 zeros.

No shared memory, no atomics — pure register/warp reduction over a
single q row's head_dim.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
)
import std.gpu.primitives.warp as warp
from std.utils.index import StaticTuple


comptime kPreprocBM: Int = 64
comptime kPreprocNThreads: Int = 128
comptime kPreprocNumWarps: Int = kPreprocNThreads // 32  # = 4
comptime kPreprocRowsPerWarp: Int = kPreprocBM // kPreprocNumWarps  # = 16


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kPreprocNThreads)
    )
)
def bwd_preprocess_kernel[
    dtype: DType,
    head_dim: Int,
](
    seq_len: Int,
    nheads: Int,
    dout_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    delta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dqaccum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    do_b_stride: Int,
    do_l_stride: Int,
    do_h_stride: Int,
    o_b_stride: Int,
    o_l_stride: Int,
    o_h_stride: Int,
    delta_b_stride: Int,
    delta_h_stride: Int,
    dqaccum_b_stride: Int,
    dqaccum_h_stride: Int,
    dqaccum_l_stride: Int,
):
    comptime cols_per_lane: Int = head_dim // 32
    comptime assert head_dim % 32 == 0, (
        "bwd_preprocess kernel currently requires head_dim % 32 == 0"
        " (32/64/128 all qualify)."
    )

    var tid: UInt32 = UInt32(thread_idx.x)
    var warp_id_v: UInt32 = tid // UInt32(WARP_SIZE)
    var lane: UInt32 = UInt32(lane_id())

    var q_tile_idx: UInt32 = UInt32(block_idx.x)
    var head_idx: UInt32 = UInt32(block_idx.y)
    var batch: UInt32 = UInt32(block_idx.z)

    var q_tile_row_base: Int = Int(q_tile_idx) * Int(kPreprocBM)
    var warp_row_base: Int = q_tile_row_base + Int(warp_id_v) * Int(
        kPreprocRowsPerWarp
    )

    var do_bh_base: Int = (
        Int(batch) * do_b_stride + Int(head_idx) * do_h_stride
    )
    var o_bh_base: Int = Int(batch) * o_b_stride + Int(head_idx) * o_h_stride
    var delta_bh_base: Int = (
        Int(batch) * delta_b_stride + Int(head_idx) * delta_h_stride
    )

    # Each warp iterates over its 16 rows; all 32 lanes participate in
    # the per-row dot product reduction.
    comptime for r in range(kPreprocRowsPerWarp):
        var q_row: Int = warp_row_base + r
        if q_row < seq_len:
            var do_row_base: Int = do_bh_base + q_row * do_l_stride
            var o_row_base: Int = o_bh_base + q_row * o_l_stride

            var acc: Float32 = Float32(0)
            # Each lane owns cols_per_lane elements at columns
            # lane, lane + 32, lane + 64, ... in the row.
            comptime for c in range(cols_per_lane):
                var col: Int = Int(lane) + c * 32
                # head_dim is a multiple of 32 and col < head_dim by
                # construction (c in [0, head_dim/32), lane in [0, 32)).
                var do_v: Float32 = (dout_ptr + do_row_base + col)[0].cast[
                    DType.float32
                ]()
                var o_v: Float32 = (o_ptr + o_row_base + col)[0].cast[
                    DType.float32
                ]()
                acc = acc + do_v * o_v

            # Warp-wide sum reduction (result available on lane 0).
            var row_sum: Float32 = warp.sum(acc)

            if lane == UInt32(0):
                (delta_ptr + delta_bh_base + q_row)[0] = row_sum

    # Zero-fill the dQaccum slice at [batch, head_idx, q_tile_row_base
    # : q_tile_row_base + BM, :]. All 128 threads cooperate over the
    # BM*head_dim fp32 elements with stride 128, so each thread writes
    # head_dim/2 zeros (head_dim is a multiple of 32 ⇒ also even). Rows
    # past seq_len in the tile-tail are skipped via the bounds check.
    var dqaccum_bh_base: Int = (
        Int(batch) * dqaccum_b_stride + Int(head_idx) * dqaccum_h_stride
    )
    comptime total_elts: Int = kPreprocBM * head_dim
    comptime elts_per_thread: Int = total_elts // kPreprocNThreads
    for i in range(elts_per_thread):
        var flat: Int = i * Int(kPreprocNThreads) + Int(tid)
        var row_in_tile: Int = flat // head_dim
        var col: Int = flat - row_in_tile * head_dim
        var q_row: Int = q_tile_row_base + row_in_tile
        if q_row < seq_len:
            (
                dqaccum_ptr
                + dqaccum_bh_base
                + q_row * dqaccum_l_stride
                + col
            )[0] = Float32(0)
