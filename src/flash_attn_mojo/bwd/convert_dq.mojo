"""Backward convert-dQ kernel: cast fp32 dqaccum -> dtype and write to dq.

Reads the fp32 `dqaccum[B, H, L, D]` workspace that the main bwd kernel
atomically accumulates dQ contributions into, casts each element down to
the output dtype (bf16 or fp16), and writes it to the user's
`dq[B, L, H, D]` tensor — note the layout transpose (H and L swapped).
Mirrors Tri Dao's `flash_bwd_convert_dq_kernel` / `convert_dQ`.

Tile config:
    BM         = 64 query rows per block
    num_threads= 128 (4 warps x 32 lanes)
    grid       = (num_m_blocks, B, H)

Each block owns the (BM, D) slice `dqaccum[b, h, q_tile_row_base:..., :]`
and writes the matching `dq[b, q_tile_row_base:..., h, :]` slice. The
128 threads strip-mine the BM*D fp32 reads / dtype writes with stride
128 (each thread handles (BM*D)/128 elements: 16 at D=32, 32 at D=64,
64 at D=128). Rows past `seq_len` are skipped via the bounds check.

No shared memory, no atomics — pure per-element cast.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    thread_idx,
)
from std.utils.index import StaticTuple


comptime kConvertBM: Int = 64
comptime kConvertNThreads: Int = 128


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kConvertNThreads)
    )
)
def bwd_convert_dq_kernel[
    dtype: DType,
    head_dim: Int,
](
    seq_len: Int,
    dqaccum_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dq_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dqaccum_b_stride: Int,
    dqaccum_h_stride: Int,
    dqaccum_l_stride: Int,
    dq_b_stride: Int,
    dq_l_stride: Int,
    dq_h_stride: Int,
):
    comptime assert (kConvertBM * head_dim) % kConvertNThreads == 0, (
        "BM*head_dim must be a multiple of num_threads"
    )

    var tid: UInt32 = UInt32(thread_idx.x)
    var q_tile_idx: UInt32 = UInt32(block_idx.x)
    var batch: UInt32 = UInt32(block_idx.y)
    var head_idx: UInt32 = UInt32(block_idx.z)

    var q_tile_row_base: Int = Int(q_tile_idx) * Int(kConvertBM)

    var dqaccum_bh_base: Int = (
        Int(batch) * dqaccum_b_stride + Int(head_idx) * dqaccum_h_stride
    )
    var dq_bh_base: Int = (
        Int(batch) * dq_b_stride + Int(head_idx) * dq_h_stride
    )

    comptime total_elts: Int = kConvertBM * head_dim
    comptime elts_per_thread: Int = total_elts // kConvertNThreads
    for i in range(elts_per_thread):
        var flat: Int = i * Int(kConvertNThreads) + Int(tid)
        var row_in_tile: Int = flat // head_dim
        var col: Int = flat - row_in_tile * head_dim
        var q_row: Int = q_tile_row_base + row_in_tile
        if q_row < seq_len:
            var src_idx: Int = (
                dqaccum_bh_base + q_row * dqaccum_l_stride + col
            )
            var v: Float32 = (dqaccum_ptr + src_idx)[0]
            var dst_idx: Int = dq_bh_base + q_row * dq_l_stride + col
            (dq_ptr + dst_idx)[0] = v.cast[dtype]()
