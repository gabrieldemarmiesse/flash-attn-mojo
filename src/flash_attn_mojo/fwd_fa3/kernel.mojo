"""Flash-attention forward kernel — Hopper FA3 path (sm_90+).

Status: SCAFFOLD ONLY. Compiles and launches but writes nothing —
the kernel body returns early. Real TMA + WGMMA implementation lands
in a follow-up commit.

Planned algorithm (matches Tri Dao FA3 fwd):

  Grid: (ceildiv(seqlen, BM), nheads, batch)

  Single warpgroup MVP (128 threads):
    1. TMA load Q tile (one-shot)
    2. For each KV block:
       a. TMA load K, V tile
       b. wait(mbarrier)
       c. WGMMA: S = Q · Kᵀ  (fp32 c-frag)
       d. online softmax: P = exp(S - max), update running scale on O
       e. WGMMA: O += P · V  (fp32 c-frag)
    3. Write O to gmem (st_matrix epilogue)

  Future (warp-specialized, 384 threads):
    - Producer warpgroup (128t): TMA loads only
    - Consumer warpgroups (256t): WGMMA + softmax + epilogue
    - Producer/consumer sync via mbarrier ping-pong
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    thread_idx,
)
from std.utils.index import StaticTuple

from common import kFa3NThreads


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kFa3NThreads))
)
def fwd_fa3_kernel[
    dtype: DType,
    head_dim: Int,
](
    seq_len: Int,
    nheads: Int,
    softmax_scale: Float32,
    q_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    q_b_stride: Int,
    q_l_stride: Int,
    q_h_stride: Int,
    k_b_stride: Int,
    k_l_stride: Int,
    k_h_stride: Int,
    v_b_stride: Int,
    v_l_stride: Int,
    v_h_stride: Int,
    o_b_stride: Int,
    o_l_stride: Int,
    o_h_stride: Int,
):
    # Scaffold body: no-op. The grid still launches so we can verify
    # the JIT + Python plumbing end-to-end. Replace with the real
    # TMA + WGMMA loop in the next commit.
    _ = q_ptr
    _ = k_ptr
    _ = v_ptr
    _ = o_ptr
    _ = seq_len
    _ = nheads
    _ = softmax_scale
    _ = q_b_stride
    _ = q_l_stride
    _ = q_h_stride
    _ = k_b_stride
    _ = k_l_stride
    _ = k_h_stride
    _ = v_b_stride
    _ = v_l_stride
    _ = v_h_stride
    _ = o_b_stride
    _ = o_l_stride
    _ = o_h_stride
    _ = block_idx.x
    _ = block_idx.y
    _ = block_idx.z
    _ = thread_idx.x
