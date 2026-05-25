"""Flash-attention forward kernel — Hopper FA3 path (sm_90+).

Single-warpgroup MVP: TMA loads of Q (one-shot), K, V (per-iter) plus
two WGMMAs per KV tile (S = Q·Kᵀ, O += P·V) with an online softmax
in between. No producer/consumer warp specialization yet — that's a
follow-up. No causal/MQA/softcap/alibi/window/dropout/LSE.

Grid: (ceildiv(seqlen, BM), nheads, batch).
Block: 1 warpgroup = 128 threads.

Algorithm per block (b, h, q_block):
  1. TMA-load Q tile at coords (b*L + q_block*BM, h, 0) -> Q_smem
  2. Init online-softmax state: rowmax = -inf, rowsum = 0, O_reg = 0
  3. For n_block in range(num_kv_blocks):
       a. TMA-load K tile at (b*L + n_block*BN, h, 0), wait
       b. wgmma S = Q · Kᵀ        (BM × BN, fp32 c-frag)
       c. online-softmax: update rowmax_new, scale prev O_reg by
          exp2(rowmax_old - rowmax_new), compute P = exp2(S - rowmax_new)
       d. TMA-load V tile at (b*L + n_block*BN, h, 0), wait
       e. wgmma O += P · V         (BM × D, fp32 c-frag)
  4. Normalize O_reg by rowsum
  5. Cast to bf16 and store to gmem
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
    WARP_SIZE,
)
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

from common import kFa3NThreads, kFa3BlockM, kFa3BlockN


# WGMMA shape choice — m64n128k16 is the largest bf16 shape that pairs
# with BM=BN=128 cleanly: 2 M-iters × 1 N-iter × 4 K-iters per QK MMA.
# For PV (depth=64, K=BN=128): m64n64k16 gives 2 M-iters × 1 N-iter × 8
# K-iters into a (BM=128, D=64) output tile.
comptime WGMMA_M: Int = 64
comptime WGMMA_N_QK: Int = 128
comptime WGMMA_N_PV: Int = 64
comptime WGMMA_K: Int = 16


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kFa3NThreads))
)
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma, `nvvm.grid_constant`)
def fwd_fa3_kernel[
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
    nheads: Int,
    softmax_scale: Float32,
    o_b_stride: Int,
    o_l_stride: Int,
    o_h_stride: Int,
):
    comptime BM: Int = kFa3BlockM
    comptime BN: Int = kFa3BlockN
    comptime D: Int = head_dim
    comptime accum_type: DType = DType.float32
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # ---- Smem layouts. K-major for both A and B (transpose_b for QK).
    comptime q_smem_layout = tile_layout_k_major[
        dtype, BM, D, swizzle_mode=swizzle
    ]()
    comptime k_smem_layout = tile_layout_k_major[
        dtype, BN, D, swizzle_mode=swizzle
    ]()
    # V is read by the PV wgmma (transpose_b=False); MN-major matches
    # the WGMMA descriptor expectation. Underlying byte layout is the
    # same as K-major (BN, D) — the TMA loads into the same memory.
    comptime v_smem_layout = tile_layout_mn_major[
        dtype, D, BN, swizzle_mode=swizzle
    ]()

    # ---- Dynamic smem carve-up.
    # Dynamic smem (must use external_memory because Q+K+V = 48 KiB
    # exactly hits the static-smem cap; mbarriers push over). Aligned
    # to 128 bytes for TMA; the +q_smem_size and +k_smem_size offsets
    # are multiples of 128 bytes so K and V stay aligned.
    comptime q_smem_size: Int = q_smem_layout.size()
    comptime k_smem_size: Int = k_smem_layout.size()

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
    var k_smem = LayoutTensor[
        dtype,
        k_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](smem_base + q_smem_size)
    var v_smem = LayoutTensor[
        dtype,
        v_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](smem_base + q_smem_size + k_smem_size)

    # ---- mbarriers: one for Q, one for K, one for V.
    var mbar_q = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var mbar_k = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var mbar_v = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    if thread_idx.x == 0:
        mbar_q[0].init()
        mbar_k[0].init()
        mbar_v[0].init()
    barrier()

    # ---- WGMMA operators.
    var wgmma_qk = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, WGMMA_N_QK, WGMMA_K),
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        transpose_b=True,
    ]()
    # PV: A comes from registers, B is V (mn-major in smem). A doesn't
    # need swizzling (regs don't use the smem swizzle); B uses the
    # configured TMA swizzle.
    var wgmma_pv = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, WGMMA_N_PV, WGMMA_K),
        a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=swizzle,
        transpose_b=False,
    ]()

    comptime num_m_mmas_qk: Int = BM // WGMMA_M  # 2
    comptime num_n_mmas_qk: Int = BN // WGMMA_N_QK  # 1
    comptime num_m_mmas_pv: Int = BM // WGMMA_M  # 2
    comptime num_n_mmas_pv: Int = D // WGMMA_N_PV  # 1

    # c-frag size per WGMMA per warpgroup thread = m*n/128.
    comptime c_frag_size_qk: Int = WGMMA_M * WGMMA_N_QK // 128  # 64
    comptime c_frag_size_pv: Int = WGMMA_M * WGMMA_N_PV // 128  # 32

    # ---- Register tiles.
    var s_reg = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas_qk * num_n_mmas_qk, c_frag_size_qk),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    var o_reg = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas_pv * num_n_mmas_pv, c_frag_size_pv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    _ = o_reg.fill(0)

    # Per-row online-softmax state. Each WGMMA c-frag has 2 rows per
    # m-iter (m16n8 sub-fragments stacked), 8 sub-row-pairs per warp.
    # We hold one rowmax/rowsum per (m_mma, row_in_mma=0/1, sub_row=0/1).
    # Total = num_m_mmas * 4 rows per thread.
    comptime rows_per_thread: Int = num_m_mmas_qk * 2 * 2  # 2 * 2 * 2 = 8
    var rowmax = stack_allocation[
        rows_per_thread, Scalar[accum_type]
    ]()
    var rowsum = stack_allocation[
        rows_per_thread, Scalar[accum_type]
    ]()

    var neg_inf: Scalar[accum_type] = Scalar[accum_type](-1.0e30)
    for i in range(rows_per_thread):
        rowmax[i] = neg_inf
        rowsum[i] = Scalar[accum_type](0)

    # ---- Coordinates for TMA loads.
    var q_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var q_row_base: Int = b_idx * seq_len + q_block * BM

    # ---- One-shot TMA load of Q.
    if thread_idx.x == 0:
        mbar_q[0].expect_bytes(Int32(BM * D * size_of[dtype]()))
        q_tma.async_copy_3d(
            q_smem,
            mbar_q[0],
            (0, h_idx, q_row_base),
        )
    barrier()

    var phase_q: UInt32 = 0
    mbar_q[0].wait(phase_q)
    phase_q ^= 1


    # ---- KV pipeline loop.
    var num_kv_blocks: Int = (seq_len + BN - 1) // BN
    var scale_log2: Scalar[accum_type] = (
        softmax_scale * Scalar[DType.float32](log2e)
    ).cast[accum_type]()

    var phase_k: UInt32 = 0
    var phase_v: UInt32 = 0

    for n_block in range(num_kv_blocks):
        var kv_row_base: Int = b_idx * seq_len + n_block * BN

        # K TMA load.
        if thread_idx.x == 0:
            mbar_k[0].expect_bytes(Int32(BN * D * size_of[dtype]()))
            k_tma.async_copy_3d(
                k_smem, mbar_k[0], (0, h_idx, kv_row_base)
            )
        barrier()
        mbar_k[0].wait(phase_k)
        phase_k ^= 1

        # WGMMA: S = Q · Kᵀ.
        _ = s_reg.fill(0)
        warpgroup_fence(s_reg)
        wgmma_qk.arrive()
        wgmma_qk.wgmma(q_smem, k_smem, s_reg)
        wgmma_qk.commit_group()
        wgmma_qk.wait_group()
        warpgroup_fence(s_reg)

        # ---- Online softmax (per c-frag row).
        # m16n8 c-frag rows: thread holds 2 rows separated by 8 (top
        # at lane/4, bot at lane/4+8). For m64n128 wgmma, num c-frags
        # per warp = 4 (BM/16 * 1 for m-direction? actually m64 means
        # 4 m16 subtiles per warpgroup spread across 4 warps; each warp
        # owns 1 m16 row group). num_n_mmas_qk=1 with WGMMA_N_QK=128.
        # Each warp owns (16, 128) of S; in c_frag_size=64 elts/thread,
        # = 16 rows × 4 col-pairs per thread? Let me think...
        #
        # Actually wgmma m64n128 produces a 64×128 tile per warpgroup.
        # 128 threads, 32 fp32 elems per thread = 4096 elems / 128 =
        # WGMMA_M * WGMMA_N / 128 = 64 * 128 / 128 = 64. ✓
        # In the warpgroup layout: 4 warps split the 64 rows into 16
        # per warp. Each warp has 32 threads in 8 row-groups × 4 col-
        # pairs. Within a row group, thread holds 2 rows × 2 cols × N
        # n-iters. Per warp: 16 rows × WGMMA_N. Per thread: 2 rows ×
        # (WGMMA_N / 8 col-pairs * 2 cols) = 2 × WGMMA_N/4 elements.
        # For WGMMA_N=128: 2 × 32 = 64 elts. ✓
        #
        # For online softmax we walk each thread's 64 fp32 c-frag and
        # treat them as (m_mma, row_in_warp=0/1, col_pairs=16). Each
        # row contributes its own rowmax/rowsum. We have 2 m_mma * 2
        # rows = 4 rows per thread (PER m_mma * top/bot). Wait num_m
        # _mmas_qk = BM/WGMMA_M = 128/64 = 2, so 2 m_mmas. Per m_mma,
        # 2 rows in the c-frag = 4 rows total per thread.
        #
        # But rows_per_thread = num_m_mmas_qk * 2 * 2 = 8 — was that
        # right? Let me recount: m16n8 sub-fragment has 4 c-frags = 2
        # row-groups × 2 col-pairs. Each row-group has 1 row in the
        # top half + 1 in the bottom half. m64 wgmma = 4 stacked m16
        # subtiles per warpgroup spread across 4 warps. So each warp
        # owns ONE m16 subtile per WGMMA_M=16-row chunk. With
        # WGMMA_M=64: 4 warps × 16 rows = 64 ✓. Each warp has 2 rows
        # × 16 col-pairs per c-frag, summed across the n direction.
        #
        # For now I'll accept this might be wrong and revisit when
        # correctness fails.

        # TODO: implement proper c-frag → (row, col) mapping for
        # online softmax. Placeholder: scale S in-place by scale_log2.
        # This produces wrong attention but exercises the wgmma path.
        for i in range(num_m_mmas_qk * num_n_mmas_qk * c_frag_size_qk):
            s_reg.ptr[i] = s_reg.ptr[i] * scale_log2

        # V TMA load.
        if thread_idx.x == 0:
            mbar_v[0].expect_bytes(Int32(BN * D * size_of[dtype]()))
            v_tma.async_copy_3d(
                v_smem, mbar_v[0], (0, h_idx, kv_row_base)
            )
        barrier()
        mbar_v[0].wait(phase_v)
        phase_v ^= 1

        # WGMMA: O += P · V, where P = exp2(S) (placeholder, no proper
        # softmax). The PV wgmma expects A as a register tile shaped
        # (num_m_mmas * num_k_mmas, a_frag_size) where a_frag_size =
        # WGMMA_M * WGMMA_K / 128. For BM=128, BN=128, that's (16, 8).
        # Underlying data is the same 128 fp32 c-frag values from S,
        # just regrouped (correctness of this reshape vs the cutlass
        # convert_layout_acc_Aregs shuffle is the next thing to verify).
        comptime a_frag_size_pv: Int = WGMMA_M * WGMMA_K // 128  # 8
        comptime num_k_mmas_pv: Int = BN // WGMMA_K              # 8
        var p_reg = LayoutTensor[
            dtype,
            Layout.row_major(num_m_mmas_pv * num_k_mmas_pv, a_frag_size_pv),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ].stack_allocation()
        for i in range(num_m_mmas_qk * num_n_mmas_qk * c_frag_size_qk):
            p_reg.ptr[i] = exp2(s_reg.ptr[i]).cast[dtype]()

        warpgroup_fence(o_reg)
        wgmma_pv.arrive()
        wgmma_pv.wgmma(p_reg, v_smem, o_reg)
        wgmma_pv.commit_group()
        wgmma_pv.wait_group()
        warpgroup_fence(o_reg)

    # ---- Output: write O_reg → gmem row-by-row.
    # m64n64 wgmma c-frag layout per warpgroup thread:
    #   warp_id (0..3) selects 16-row slab within BM=64 per WGMMA_M.
    #   lane (0..31) splits as group=lane/4 (row 0..7) + pair=lane%4
    #     (col-pair 0..3 ⇒ cols 0,2,4,6 + 1,3,5,7 ⇒ 8 cols per pair).
    # Per thread elements: m_mma × n_mma × c_frag_size_pv = 2×1×32 = 64.
    #   Within one c_frag of 32 elts: 4 sub-pairs (rows 0..7) × top/bot
    #   (row +8) × 2 cols × num_n_mma_sub (WGMMA_N=64 / 8 = 8 sub).
    #
    # Placeholder simple write: each thread writes its 64 fp32 c-frag
    # values cast to bf16 at row=warp*16 + group + {0,8}, with
    # col=n_sub*8 + pair*2 + {0,1}. This won't produce correct O yet
    # because the softmax is a placeholder.
    var lane: Int = Int(lane_id())
    var w_id: Int = Int(warp_id())
    var lane_group: Int = lane // 4
    var lane_pair: Int = lane % 4

    var o_base: Int = (
        b_idx * o_b_stride + h_idx * o_h_stride + q_block * BM * o_l_stride
    )

    comptime for m_mma in range(num_m_mmas_pv):
        # Each WGMMA m=64 spreads 16 rows per warp. m_mma indexes
        # which WGMMA_M chunk along BM.
        var row_warp_base: Int = m_mma * WGMMA_M + w_id * 16
        comptime for n_mma in range(num_n_mmas_pv):
            var col_base: Int = n_mma * WGMMA_N_PV
            comptime for n_sub in range(WGMMA_N_PV // 8):
                # 8 cols per sub-frag, 4 elts per thread per sub-frag.
                var c_off: Int = (
                    (m_mma * num_n_mmas_pv + n_mma) * c_frag_size_pv
                    + n_sub * 4
                )
                var row_top: Int = row_warp_base + lane_group
                var row_bot: Int = row_top + 8
                var col_a: Int = col_base + n_sub * 8 + 2 * lane_pair
                var col_b: Int = col_a + 1

                # No normalization (placeholder).
                var c0 = o_reg.ptr[c_off + 0]
                var c1 = o_reg.ptr[c_off + 1]
                var c2 = o_reg.ptr[c_off + 2]
                var c3 = o_reg.ptr[c_off + 3]

                if row_top < seq_len - q_block * BM:
                    (o_ptr + o_base + row_top * o_l_stride + col_a)[0] = (
                        c0.cast[dtype]()
                    )
                    (o_ptr + o_base + row_top * o_l_stride + col_b)[0] = (
                        c1.cast[dtype]()
                    )
                if row_bot < seq_len - q_block * BM:
                    (o_ptr + o_base + row_bot * o_l_stride + col_a)[0] = (
                        c2.cast[dtype]()
                    )
                    (o_ptr + o_base + row_bot * o_l_stride + col_b)[0] = (
                        c3.cast[dtype]()
                    )
