"""FA4-target flash-attention forward kernel (sm_90a, Hopper).

v2: + 2-stage K/V smem pipeline (TMA for tile n+1 issued while tile n
computes), QK wgmma accumulates with scale_c=0 (no register-tile
zero-fill), max.f32 softmax reductions, reciprocal normalization,
paired output stores. Still 2 MMA warpgroups / 256 threads, no
producer warpgroup, no intra-warpgroup overlap.

Grid: (ceildiv(seqlen, BM), nheads, batch).
Block: 256 threads = 2 warpgroups, each owning a 64-row half of the
(BM=128, D=128) Q tile.

Per block (b, h, m_block):
  1. TMA-load Q tile once.
  2. issue TMA K/V loads of tile 0 into stage 0.
  3. For n_block:
       issue TMA K/V loads of tile n+1 into stage (n+1)%2
       wait K[n%2];  S = Q·Kᵀ        (wgmma m64n128k16, SS, scale_c=0)
       online softmax (exp2 trick), P = exp2(S·scale - m)
       wait V[n%2];  O += P·V        (wgmma m64n128k16, RS)
       block-wide barrier (stage n%2 buffers become writable)
  4. O *= 1/rowsum, cast bf16, store to gmem as 32-bit pairs.

P c-frag -> a-frag mapping: with num_m_mmas=1 per warpgroup the QK
c-fragment element order (16 col-chunks x [top0 top1 bot0 bot1]) is
identical to the PV a-fragment order (8 k_mmas x 8 halves) — a
straight indexwise cast is correct. (With >1 m_mma it would not be:
the RS wgmma walks fragments k-major, `a_frags[m + k*num_m]`.)
"""

from std.math import exp2
from std.math.constants import log2e
from std.sys import size_of
from std.utils.index import StaticTuple, IndexList

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
import std.gpu.primitives.warp as warp
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import AddressSpace, external_memory
from std.memory import stack_allocation

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from common import kFa4NThreads, kFa4BlockM, kFa4BlockN, kFa4NMmaWarpgroups

comptime WGMMA_M: Int = 64
comptime WGMMA_K: Int = 16
comptime NSTAGES: Int = 2


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kFa4NThreads))
)
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma, `nvvm.grid_constant`)
def fwd_fa4_kernel[
    dtype: DType,
    head_dim: Int,
    q_tile_shape: IndexList[3],
    q_desc_shape: IndexList[3],
    kv_tile_shape: IndexList[3],
    kv_desc_shape: IndexList[3],
](
    q_tma: TMATensorTile[dtype, 3, q_tile_shape, q_desc_shape],
    k_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    v_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_len: Int,
    softmax_scale: Float32,
    o_b_stride: Int,
    o_l_stride: Int,
    o_h_stride: Int,
):
    comptime BM: Int = kFa4BlockM
    comptime BN: Int = kFa4BlockN
    comptime D: Int = head_dim
    comptime NWG: Int = kFa4NMmaWarpgroups
    comptime accum_type: DType = DType.float32
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # ---- Smem layouts (K-major for Q/K; V as MN-major view of the
    # same row-major (BN, D) bytes the TMA writes).
    comptime q_smem_layout = tile_layout_k_major[
        dtype, BM, D, swizzle_mode=swizzle
    ]()
    comptime k_smem_layout = tile_layout_k_major[
        dtype, BN, D, swizzle_mode=swizzle
    ]()
    comptime v_smem_layout = tile_layout_mn_major[
        dtype, D, BN, swizzle_mode=swizzle
    ]()

    comptime q_smem_size: Int = q_smem_layout.size()
    comptime k_smem_size: Int = k_smem_layout.size()
    comptime v_smem_size: Int = v_smem_layout.size()

    var smem_base = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]()
    var q_smem = LayoutTensor[
        dtype,
        q_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](smem_base)
    var k_smem_base = smem_base + q_smem_size
    var v_smem_base = k_smem_base + NSTAGES * k_smem_size

    # ---- mbarriers (one per stage per tensor + one for Q).
    var mbar_q = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var mbar_k = stack_allocation[
        NSTAGES, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var mbar_v = stack_allocation[
        NSTAGES, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    if thread_idx.x == 0:
        mbar_q[0].init()
        comptime for s in range(NSTAGES):
            mbar_k[s].init()
            mbar_v[s].init()
    barrier()

    # ---- WGMMA operators. Both GEMMs are m64 n128 k16.
    var wgmma_qk = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, BN, WGMMA_K),
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        transpose_b=True,
    ]()
    var wgmma_pv = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, D, WGMMA_K),
        a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=swizzle,
        transpose_b=False,
    ]()

    # Per-warpgroup fragment sizes (num_m_mmas = num_n_mmas = 1).
    comptime c_frag_size_qk: Int = WGMMA_M * BN // 128  # 64
    comptime c_frag_size_pv: Int = WGMMA_M * D // 128  # 64
    comptime a_frag_size_pv: Int = WGMMA_M * WGMMA_K // 128  # 8
    comptime num_k_mmas_pv: Int = BN // WGMMA_K  # 8

    var s_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_size_qk),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var o_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_size_pv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    _ = o_reg.fill(0)
    var p_reg = LayoutTensor[
        dtype,
        Layout.row_major(num_k_mmas_pv, a_frag_size_pv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    # Online-softmax state: 2 rows per thread (top/bot of the m64).
    comptime rows_per_thread: Int = 2
    var rowmax = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var rowsum = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var neg_inf: Scalar[accum_type] = Scalar[accum_type](-1.0e30)
    for i in range(rows_per_thread):
        rowmax[i] = neg_inf
        rowsum[i] = Scalar[accum_type](0)

    var wg: Int = Int(thread_idx.x) // 128

    # ---- TMA coordinates.
    var m_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var q_row_base: Int = b_idx * seq_len + m_block * BM

    var num_kv_blocks: Int = (seq_len + BN - 1) // BN
    var scale_log2: Scalar[accum_type] = (
        softmax_scale * Scalar[DType.float32](log2e)
    ).cast[accum_type]()

    @parameter
    @always_inline
    def issue_kv_load(n_block: Int, stage: Int):
        """Thread 0 only: arm + issue the K and V TMA loads of tile
        `n_block` into smem stage `stage`."""
        var kv_row_base: Int = b_idx * seq_len + n_block * BN
        var k_st = LayoutTensor[
            dtype,
            k_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](k_smem_base + stage * k_smem_size)
        var v_st = LayoutTensor[
            dtype,
            v_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](v_smem_base + stage * v_smem_size)
        mbar_k[stage].expect_bytes(Int32(BN * D * size_of[dtype]()))
        k_tma.async_copy_3d(k_st, mbar_k[stage], (0, h_idx, kv_row_base))
        mbar_v[stage].expect_bytes(Int32(BN * D * size_of[dtype]()))
        v_tma.async_copy_3d(v_st, mbar_v[stage], (0, h_idx, kv_row_base))

    # ---- One-shot Q load + first K/V tile.
    if thread_idx.x == 0:
        mbar_q[0].expect_bytes(Int32(BM * D * size_of[dtype]()))
        q_tma.async_copy_3d(q_smem, mbar_q[0], (0, h_idx, q_row_base))
        issue_kv_load(0, 0)
    mbar_q[0].wait(UInt32(0))

    var phase_k = stack_allocation[NSTAGES, UInt32]()
    var phase_v = stack_allocation[NSTAGES, UInt32]()
    comptime for s in range(NSTAGES):
        phase_k[s] = 0
        phase_v[s] = 0

    for n_block in range(num_kv_blocks):
        var stage: Int = n_block % NSTAGES

        # Issue the next tile's loads into the other stage. Its
        # buffers were released by the barrier ending iter n-1.
        if thread_idx.x == 0 and n_block + 1 < num_kv_blocks:
            issue_kv_load(n_block + 1, (n_block + 1) % NSTAGES)

        var k_smem = LayoutTensor[
            dtype,
            k_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](k_smem_base + stage * k_smem_size)
        var v_smem = LayoutTensor[
            dtype,
            v_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](v_smem_base + stage * v_smem_size)

        mbar_k[stage].wait(phase_k[stage])
        phase_k[stage] ^= 1

        # S = Q · Kᵀ for this warpgroup's 64 rows (scale_c=0: first
        # k-step overwrites, no zero-fill needed).
        warpgroup_fence(s_reg)
        wgmma_qk.arrive()
        wgmma_qk.wgmma[num_warp_groups=NWG, scale_c=0](
            q_smem, k_smem, s_reg, wg
        )
        wgmma_qk.commit_group()
        wgmma_qk.wait_group()
        warpgroup_fence(s_reg)

        # ---- Online softmax (base-2 exp). c-frag row of element c is
        # top (c%4 < 2) or bottom (c%4 >= 2).
        comptime for c in range(c_frag_size_qk):
            s_reg.ptr[c] = s_reg.ptr[c] * scale_log2

        var local_max = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        for i in range(rows_per_thread):
            local_max[i] = neg_inf
        comptime for c in range(c_frag_size_qk):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            local_max[row_idx] = max(local_max[row_idx], s_reg.ptr[c])
        @parameter
        for i in range(rows_per_thread):
            local_max[i] = warp.lane_group_max[num_lanes=4](local_max[i])

        var scale_old = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        @parameter
        for i in range(rows_per_thread):
            var rmax_new: Scalar[accum_type] = max(local_max[i], rowmax[i])
            scale_old[i] = exp2(rowmax[i] - rmax_new)
            rowmax[i] = rmax_new
            rowsum[i] *= scale_old[i]

        comptime for c in range(c_frag_size_pv):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            o_reg.ptr[c] *= scale_old[row_idx]

        var local_sum = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        for i in range(rows_per_thread):
            local_sum[i] = Scalar[accum_type](0)
        comptime for c in range(c_frag_size_qk):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            var p: Scalar[accum_type] = exp2(s_reg.ptr[c] - rowmax[row_idx])
            s_reg.ptr[c] = p
            local_sum[row_idx] += p
        @parameter
        for i in range(rows_per_thread):
            rowsum[i] += local_sum[i]

        # P: straight indexwise cast (valid for num_m_mmas=1, see top).
        comptime for c in range(c_frag_size_qk):
            p_reg.ptr[c] = s_reg.ptr[c].cast[dtype]()

        mbar_v[stage].wait(phase_v[stage])
        phase_v[stage] ^= 1

        warpgroup_fence(o_reg)
        wgmma_pv.arrive()
        wgmma_pv.wgmma(p_reg, v_smem, o_reg)
        wgmma_pv.commit_group()
        wgmma_pv.wait_group()
        warpgroup_fence(o_reg)

        # Release this stage's buffers: every thread is past its
        # wgmma reads before thread 0 reuses them for tile n+2.
        barrier()

    # ---- rowsum: reduce across the 4 lanes sharing each row, then
    # normalize with a reciprocal (one div per row, not per element).
    var inv_rowsum = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    @parameter
    for i in range(rows_per_thread):
        rowsum[i] = warp.lane_group_sum[num_lanes=4](rowsum[i])
        inv_rowsum[i] = Scalar[accum_type](1) / rowsum[i]

    comptime for c in range(c_frag_size_pv):
        comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
        o_reg.ptr[c] *= inv_rowsum[row_idx]

    # ---- Store O. c-frag walk: col_chunk = c/4, in_chunk = c%4,
    # row = wg*64 + warp*16 + lane/4 (+8 if bottom),
    # col = col_chunk*8 + 2*(lane%4) + (in_chunk&1).
    # in_chunk pairs (0,1) and (2,3) are contiguous columns — store
    # them as one 2-element vector each.
    var lane: Int = Int(lane_id())
    var warp_in_wg: Int = Int(warp_id()) % 4
    var lane_group: Int = lane // 4
    var lane_pair: Int = lane % 4

    var o_base: Int = (
        b_idx * o_b_stride + h_idx * o_h_stride + m_block * BM * o_l_stride
    )
    var rows_in_block: Int = seq_len - m_block * BM
    var row_warp_base: Int = wg * WGMMA_M + warp_in_wg * 16

    comptime for c2 in range(c_frag_size_pv // 2):
        comptime col_chunk: Int = c2 // 2
        comptime is_bot: Int = c2 % 2
        var row: Int = row_warp_base + lane_group + (8 if is_bot == 1 else 0)
        var col: Int = col_chunk * 8 + 2 * lane_pair
        if row < rows_in_block:
            var pair = SIMD[dtype, 2](
                o_reg.ptr[2 * c2].cast[dtype](),
                o_reg.ptr[2 * c2 + 1].cast[dtype](),
            )
            (o_ptr + o_base + row * o_l_stride + col).store[width=2](pair)
