"""FA4-target flash-attention backward kernels (sm_90a, Hopper).

Three kernels, mirroring FA4's pipeline:

1. `bwd_preprocess_kernel`: dpsum = rowsum(dO * O) (fp32),
   lse_log2 = lse * log2(e), and zero dq_accum.
2. `bwd_main_kernel`: grid over KV tiles. Each block TMA-loads its
   K/V tile once, then loops over Q tiles (BM=64 rows), streaming
   Q/dO through a 6-slot smem ring filled by a producer warpgroup.
   Per m-tile, the 2 MMA warpgroups (each owning 64 of the 128 KV
   rows) compute, in scaled-log2 domain:

       S^T  = K · Q^T            (wgmma SS, m64n64k16, swapAB)
       dP^T = V · dO^T           (wgmma SS, m64n64k16, swapAB)
       P^T  = exp2(S^T*scale_log2 - lse_log2[col])
       dV  += P^T · dO           (wgmma RS, m64n128k16)
       dS^T = P^T * (dP^T - dpsum[col])
       dK  += dS^T · Q           (wgmma RS, m64n128k16)
       sdS  = dS (transposed store to smem, k-major (BM, BN))
       dQ  += dS · K             (wgmma SS on warpgroup 0 only,
                                  m64n128k16; B = K as mn-major view)
       dq_accum[q, d] +=atomic dQ c-frag   (fp32 gmem)

   Epilogue: dK *= softmax_scale; dV and dK staged to smem
   (16B-chunk-major) and TMA bulk-stored.
3. `bwd_convert_kernel`: dq = (dq_accum * softmax_scale).bf16.

P^T / dS^T c-frag -> RS a-frag: straight indexwise cast (valid at
num_m_mmas=1, same argument as the fwd kernel).

dq_accum layout is (B, H, S, D) fp32 so the dQ c-frag's column pairs
(adjacent d) are contiguous. dpsum/lse_log2 are (B, H, S) fp32.
All q/k/v/o/do/dq/dk/dv tensors are contiguous (B, S, H, D) bf16.
"""

from std.math import exp2
from std.math.constants import log2e
from std.sys import size_of
from std.utils.index import StaticTuple, IndexList

from std.atomic import Atomic, Ordering
from std.sys._assembly import inlined_assembly
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
import std.gpu.primitives.warp as warp
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
)
from std.gpu.sync import named_barrier, syncwarp
from std.memory import stack_allocation

from std.gpu.compute.mma import wgmma_async

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    _wgmma_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from common import (
    kBwdBlockM,
    kBwdBlockN,
    kBwdNMmaWarpgroups,
    kBwdNThreads,
    kBwdQdOStages,
    kBwdPreBlockM,
    kBwdPreThreads,
)

comptime WGMMA_M: Int = 64
comptime WGMMA_K: Int = 16
comptime NUM_PRODUCER_REGS: Int = 24
comptime NUM_CONSUMER_REGS: Int = 240


# ===================================================================
# Main backward kernel
# ===================================================================
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kBwdNThreads))
)
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(do_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(dk_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(dv_tma, `nvvm.grid_constant`)
def bwd_main_kernel[
    dtype: DType,
    head_dim: Int,
    q_tile_shape: IndexList[3],
    q_desc_shape: IndexList[3],
    kv_tile_shape: IndexList[3],
    kv_desc_shape: IndexList[3],
    st_tile_shape: IndexList[3],
    st_desc_shape: IndexList[3],
](
    q_tma: TMATensorTile[dtype, 3, q_tile_shape, q_desc_shape],
    do_tma: TMATensorTile[dtype, 3, q_tile_shape, q_desc_shape],
    k_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    v_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    dk_tma: TMATensorTile[dtype, 3, st_tile_shape, st_desc_shape],
    dv_tma: TMATensorTile[dtype, 3, st_tile_shape, st_desc_shape],
    lse_log2_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dpsum_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dq_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int,
    softmax_scale: Float32,
):
    comptime BM: Int = kBwdBlockM
    comptime BN: Int = kBwdBlockN
    comptime D: Int = head_dim
    comptime NWG: Int = kBwdNMmaWarpgroups
    comptime STAGES: Int = kBwdQdOStages
    comptime accum_type: DType = DType.float32
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # ---- smem layouts.
    comptime kv_smem_layout = tile_layout_k_major[
        dtype, BN, D, swizzle_mode=swizzle
    ]()
    # mn-major (D, BN) view of the same K bytes for the dQ GEMM's B.
    comptime kt_view_layout = tile_layout_mn_major[
        dtype, D, BN, swizzle_mode=swizzle
    ]()
    comptime q_smem_layout = tile_layout_k_major[
        dtype, BM, D, swizzle_mode=swizzle
    ]()
    # mn-major (D, BM) view of a Q/dO slot for the dK/dV GEMMs' B.
    comptime qt_view_layout = tile_layout_mn_major[
        dtype, D, BM, swizzle_mode=swizzle
    ]()
    # dS^T staged in its natural (BN rows, BM cols) orientation with
    # the 128B swizzle (BM=64 bf16 cols = exactly one swizzle row).
    # Viewed mn-major (BM, BN) it is the B operand of the hand-rolled
    # dQ^T GEMM below (same trick as fwd's V tile).
    comptime sds_b_layout = tile_layout_mn_major[
        dtype, BM, BN, swizzle_mode=swizzle
    ]()
    comptime sds_canonical = tile_to_descriptor[
        dtype, sds_b_layout, False
    ]()
    # A operand of dQ^T = K^T (D, BN) = the mn-major view of K's
    # bytes. TensorCoreAsync only supports k-major A, so the dQ^T
    # GEMM is hand-rolled with raw wgmma_async[layout_a="col"].
    comptime kt_canonical = tile_to_descriptor[
        dtype, kt_view_layout, False
    ]()
    comptime kt_shape00: Int = kt_canonical[0].shape[0].value()
    comptime kt_stride01: Int = kt_canonical[0].stride[1].value()
    comptime kt_stride11: Int = kt_canonical[1].stride[1].value()
    comptime a_wg_stride: Int = (
        kt_stride01 * (WGMMA_M // kt_shape00) * size_of[dtype]()
    )
    comptime a_k_stride: Int = kt_stride11 * 2 * size_of[dtype]()
    comptime sds_stride11: Int = sds_canonical[1].stride[1].value()
    comptime b_k_stride: Int = sds_stride11 * 2 * size_of[dtype]()

    comptime kv_tile_size: Int = BN * D
    comptime q_slot_size: Int = BM * D
    comptime sds_size: Int = BM * BN

    var smem_base = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]()
    var k_base = smem_base
    var v_base = k_base + kv_tile_size
    var ring_base = v_base + kv_tile_size
    # sdS is double-buffered: stage m%2 is written at iter m and read
    # by the dQ GEMM; the pre-dQ barrier of iter m+1 proves dQ(m)
    # retired before either warpgroup rewrites stage m%2 at iter m+2.
    var sds_base = ring_base + STAGES * q_slot_size
    # lse_log2/dpsum smem ring: 2 stages x BM f32 each, prefetched
    # one m-tile ahead (consumer threads issue the gmem loads; the
    # per-iter named barrier orders store->read across warpgroups).
    # lse_log2/dpsum ride the Q pipeline (FA4's sLSE design): one
    # BM-f32 buffer per Q/dO slot pair, filled by the producer warp
    # and published by the Q slot's full barrier (init(2): TMA
    # expect + producer arrive); recycled with the slot's empties.
    var lse_smem = (sds_base + 2 * sds_size).bitcast[Float32]()
    var dps_smem = lse_smem + (STAGES // 2) * BM

    var k_smem = LayoutTensor[
        dtype,
        kv_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](k_base)
    var v_smem = LayoutTensor[
        dtype,
        kv_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](v_base)
    var kt_view = LayoutTensor[
        dtype,
        kt_view_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](k_base)
    # ---- mbarriers.
    var mbar_k = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var mbar_v = stack_allocation[
        1, SharedMemBarrier, address_space=AddressSpace.SHARED, alignment=8
    ]()
    var full = stack_allocation[
        STAGES,
        SharedMemBarrier,
        address_space=AddressSpace.SHARED,
        alignment=8,
    ]()
    var empty = stack_allocation[
        STAGES,
        SharedMemBarrier,
        address_space=AddressSpace.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar_k[0].init()
        mbar_v[0].init()
        comptime for s in range(STAGES):
            # Even (Q) slots: TMA expect-arrive + producer-warp
            # arrive after staging lse/dps.
            full[s].init(2 if s % 2 == 0 else 1)
            empty[s].init(Int32(NWG * 128))
    barrier()

    var n_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var num_m_blocks: Int = seq_len // BM
    var kv_row: Int = b_idx * seq_len + n_block * BN

    var wgid: Int = Int(thread_idx.x) // 128

    if wgid == 0:
        # ================= producer =================
        warpgroup_reg_dealloc[NUM_PRODUCER_REGS]()
        if thread_idx.x < 32:
            var lane: Int = Int(thread_idx.x)
            if lane == 0:
                mbar_k[0].expect_bytes(Int32(BN * D * size_of[dtype]()))
                k_tma.async_copy_3d(k_smem, mbar_k[0], (0, h_idx, kv_row))
                mbar_v[0].expect_bytes(Int32(BN * D * size_of[dtype]()))
                v_tma.async_copy_3d(v_smem, mbar_v[0], (0, h_idx, kv_row))

            var slot: Int = 0
            var phase: UInt32 = 0
            var wrap: Int = 0
            var q_row: Int = b_idx * seq_len
            var lse_row: Int = (
                b_idx * Int(grid_dim.y) + h_idx
            ) * seq_len
            for _ in range(num_m_blocks):
                empty[slot].wait(phase)
                if lane == 0:
                    var q_st = LayoutTensor[
                        dtype,
                        q_smem_layout,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](ring_base + slot * q_slot_size)
                    full[slot].expect_bytes(
                        Int32(BM * D * size_of[dtype]())
                    )
                    q_tma.async_copy_3d(
                        q_st, full[slot], (0, h_idx, q_row)
                    )
                # Stage this tile's lse_log2/dpsum (2 f32 each per
                # lane) into the slot-pair's buffer, then publish via
                # the Q slot's second arrival.
                var lse_buf = lse_smem + (slot // 2) * BM
                var dps_buf = dps_smem + (slot // 2) * BM
                comptime for j in range(BM // 64):
                    var idx: Int = j * 64 + lane * 2
                    (lse_buf + idx).store[width=2](
                        (lse_log2_ptr + lse_row + idx).load[width=2]()
                    )
                    (dps_buf + idx).store[width=2](
                        (dpsum_ptr + lse_row + idx).load[width=2]()
                    )
                fence_async_view_proxy()
                syncwarp()
                if lane == 0:
                    _ = full[slot].arrive()

                empty[slot + 1].wait(phase)
                if lane == 0:
                    var do_st = LayoutTensor[
                        dtype,
                        q_smem_layout,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](ring_base + (slot + 1) * q_slot_size)
                    full[slot + 1].expect_bytes(
                        Int32(BM * D * size_of[dtype]())
                    )
                    do_tma.async_copy_3d(
                        do_st, full[slot + 1], (0, h_idx, q_row)
                    )

                q_row += BM
                lse_row += BM
                slot += 2
                wrap += 1
                if wrap == 3:
                    wrap = 0
                    slot = 0
                    phase ^= 1
        return

    # ================= MMA warpgroups =================
    warpgroup_reg_alloc[NUM_CONSUMER_REGS]()
    var wg: Int = wgid - 1

    comptime for s in range(STAGES):
        _ = empty[s].arrive()

    # ---- WGMMA operators.
    # S^T / dP^T: (BN x BM) = KV-tile · {Q,dO}-tile^T, M split by wg.
    var wgmma_sdp = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, BM, WGMMA_K),
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        transpose_b=True,
    ]()
    # dV / dK: (BN x D) += {P,dS}^T_regs · {dO,Q} (B = mn-major view).
    var wgmma_dkv = TensorCoreAsync[
        accum_type,
        dtype,
        dtype,
        IndexList[3](WGMMA_M, D, WGMMA_K),
        a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=swizzle,
        transpose_b=False,
    ]()

    comptime c_frag_sdp: Int = WGMMA_M * BM // 128  # 32
    comptime c_frag_dkv: Int = WGMMA_M * D // 128  # 64
    comptime a_frag: Int = WGMMA_M * WGMMA_K // 128  # 8
    comptime num_k_mmas_rs: Int = BM // WGMMA_K  # 4

    var s_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_sdp),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var dp_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_sdp),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var p_reg = LayoutTensor[
        dtype,
        Layout.row_major(num_k_mmas_rs, a_frag),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var ds_reg = LayoutTensor[
        dtype,
        Layout.row_major(num_k_mmas_rs, a_frag),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var dv_acc = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_dkv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    var dk_acc = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_dkv),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    _ = dv_acc.fill(0)
    _ = dk_acc.fill(0)
    comptime c_frag_dq: Int = WGMMA_M * BM // 128  # 32
    var dq_reg = LayoutTensor[
        accum_type,
        Layout.row_major(1, c_frag_dq),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()

    var lane: Int = Int(lane_id())
    var warp_in_wg: Int = Int(warp_id()) % 4
    var lane_group: Int = lane // 4
    var lane_pair: Int = lane % 4

    var scale_log2: Scalar[accum_type] = (
        softmax_scale * Scalar[DType.float32](log2e)
    ).cast[accum_type]()

    # Per-(b,h) row base for lse_log2 / dpsum ((B, H, S) fp32);
    # grid_dim.y == nheads. dq_accum is (B, H, D, S) fp32.
    var bh_row_base: Int = (b_idx * Int(grid_dim.y) + h_idx) * seq_len
    var dq_bh_base: Int = (b_idx * Int(grid_dim.y) + h_idx) * D

    # ---- consumer state.
    mbar_k[0].wait(UInt32(0))
    mbar_v[0].wait(UInt32(0))

    var slot: Int = 0
    var phase: UInt32 = 0
    var wrap: Int = 0
    var q_base: Int = 0
    var sds_stage: Int = 0

    for _ in range(num_m_blocks):
        var q_view = LayoutTensor[
            dtype,
            q_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](ring_base + slot * q_slot_size)
        var qt_view = LayoutTensor[
            dtype,
            qt_view_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](ring_base + slot * q_slot_size)
        var do_view = LayoutTensor[
            dtype,
            q_smem_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](ring_base + (slot + 1) * q_slot_size)
        var dot_view = LayoutTensor[
            dtype,
            qt_view_layout,
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](ring_base + (slot + 1) * q_slot_size)

        # S^T = K · Q^T
        full[slot].wait(phase)

        # lse_log2/dpsum for this tile (published with the Q slot).
        var lse_vals = stack_allocation[16, Scalar[accum_type]]()
        var dps_vals = stack_allocation[16, Scalar[accum_type]]()
        comptime for cc in range(BM // 8):  # 8 col chunks
            var col: Int = cc * 8 + 2 * lane_pair
            var lp2 = (
                lse_smem + (slot // 2) * BM + col
            ).load[width=2]()
            var dp2 = (
                dps_smem + (slot // 2) * BM + col
            ).load[width=2]()
            lse_vals[cc * 2] = lp2[0]
            lse_vals[cc * 2 + 1] = lp2[1]
            dps_vals[cc * 2] = dp2[0]
            dps_vals[cc * 2 + 1] = dp2[1]
        warpgroup_fence(s_reg)
        wgmma_sdp.arrive()
        wgmma_sdp.wgmma[num_warp_groups=NWG, scale_c=0](
            k_smem, q_view, s_reg, wg
        )
        wgmma_sdp.commit_group()

        # dP^T = V · dO^T
        full[slot + 1].wait(phase)
        warpgroup_fence(dp_reg)
        wgmma_sdp.arrive()
        wgmma_sdp.wgmma[num_warp_groups=NWG, scale_c=0](
            v_smem, do_view, dp_reg, wg
        )
        wgmma_sdp.commit_group()

        # S^T retired; dP^T still in flight.
        wgmma_sdp.wait_group[1]()
        warpgroup_fence(s_reg)

        # P^T = exp2(S^T * scale_log2 - lse_log2[q col]); P kept in
        # f32 in s_reg, packed bf16 into p_reg.
        comptime for c in range(c_frag_sdp):
            comptime cc: Int = c // 4
            comptime j: Int = c & 1
            s_reg.ptr[c] = exp2(
                s_reg.ptr[c].fma(scale_log2, -lse_vals[cc * 2 + j])
            )
        comptime for c in range(c_frag_sdp):
            p_reg.ptr[c] = s_reg.ptr[c].cast[dtype]()

        # dV += P^T · dO
        warpgroup_fence(dv_acc)
        wgmma_dkv.arrive()
        wgmma_dkv.wgmma(p_reg, dot_view, dv_acc)
        wgmma_dkv.commit_group()

        # dP^T retired; dV still in flight.
        wgmma_sdp.wait_group[1]()
        warpgroup_fence(dp_reg)

        # dS^T = P^T * (dP^T - dpsum[q col])
        comptime for c in range(c_frag_sdp):
            comptime cc: Int = c // 4
            comptime j: Int = c & 1
            dp_reg.ptr[c] = s_reg.ptr[c] * (
                dp_reg.ptr[c] - dps_vals[cc * 2 + j]
            )
        comptime for c in range(c_frag_sdp):
            ds_reg.ptr[c] = dp_reg.ptr[c].cast[dtype]()

        # Stage dS^T in its natural (kv row, q col) orientation with
        # the 128B swizzle. The hardware computes the swizzle XOR
        # from the *absolute* smem address ((addr>>7)&7), and the
        # dynamic-smem base is not necessarily 1024B-aligned (the
        # static-smem mbarriers shift it), so the base's line phase
        # is folded into the XOR: offset(r, c) =
        # r*BM + ((c/8)^((r + phase)%8))*8 + c%8. Pairs (c, c+1)
        # stay contiguous -> 32-bit stores, conflict-free.
        var sds_stage_base = sds_base + sds_stage * sds_size
        var sds_phase: Int = (Int(sds_stage_base) >> 7) & 7
        var kv_r_lo: Int = wg * WGMMA_M + warp_in_wg * 16 + lane_group
        var xor_lo: Int = (lane_group + sds_phase) & 7
        comptime for c2 in range(c_frag_sdp // 2):
            comptime cc: Int = c2 // 2
            comptime bot: Int = c2 % 2
            var r: Int = kv_r_lo + (8 if bot == 1 else 0)
            var off: Int = (
                r * BM + ((cc ^ xor_lo) * 8) + 2 * lane_pair
            )
            var pair = SIMD[dtype, 2](
                ds_reg.ptr[2 * c2], ds_reg.ptr[2 * c2 + 1]
            )
            (sds_stage_base + off).store[width=2, alignment=4](pair)

        fence_async_view_proxy()
        # Single per-iter barrier: proves both warpgroups wrote this
        # stage of sdS *and* (transitively, via last iter's
        # wait_group below) that dQ(iter-1) retired before its stage
        # gets rewritten next iteration.
        named_barrier[Int32(NWG * 128)](Int32(4))

        # dQ^T (D x BM) = K^T · dS, hand-rolled (layout_a="col" so A
        # is K's bytes read mn-major). M = D = 128 split m64 per
        # warpgroup -> both warpgroups share the GEMM.
        warpgroup_fence(dq_reg)
        wgmma_sdp.arrive()
        var a_desc = _wgmma_descriptor[kt_canonical, False, swizzle](
            k_base
        ) + wg * a_wg_stride
        var b_desc = _wgmma_descriptor[sds_canonical, False, swizzle](
            sds_stage_base
        )
        var dq_frags = dq_reg.vectorize[1, c_frag_dq]()
        comptime for k_mma in range(BN // WGMMA_K):
            dq_frags[0, 0] = wgmma_async[
                WGMMA_M,
                BM,
                WGMMA_K,
                accum_type,
                a_type=dtype,
                b_type=dtype,
                layout_a="col",
                layout_b="row",
                scale_d = 0 if k_mma == 0 else 1,
            ](
                a_desc + k_mma * a_k_stride,
                b_desc + k_mma * b_k_stride,
                dq_frags[0, 0],
            )
        wgmma_sdp.commit_group()

        # dK += dS^T · Q — committed AFTER dQ (FA4's order) so the
        # dQ drain below overlaps the dK GEMM on the tensor core.
        warpgroup_fence(dk_acc)
        wgmma_dkv.arrive()
        wgmma_dkv.wgmma(ds_reg, qt_view, dk_acc)
        wgmma_dkv.commit_group()

        # Queue [dV, dQ, dK]: wait ≤1 retires dV and dQ; dK still
        # runs while we drain dQ. dO(n) consumed by dV -> release.
        wgmma_dkv.wait_group[1]()
        warpgroup_fence(dv_acc)
        warpgroup_fence(dq_reg)
        _ = empty[slot + 1].arrive()

        # Drain dQ^T to dq_accum (B,H,D,S) under the dK GEMM.
        comptime for c2 in range(c_frag_dq // 2):
            comptime cc: Int = c2 // 2
            comptime bot: Int = c2 % 2
            var d_row: Int = (
                wg * WGMMA_M
                + warp_in_wg * 16
                + lane_group
                + (8 if bot == 1 else 0)
            )
            var qc: Int = cc * 8 + 2 * lane_pair
            inlined_assembly[
                "red.relaxed.gpu.global.add.v2.f32 [$0], {$1, $2};",
                NoneType,
                constraints="l,f,f",
            ](
                dq_accum_ptr
                + (dq_bh_base + d_row) * seq_len
                + q_base
                + qc,
                dq_reg.ptr[2 * c2],
                dq_reg.ptr[2 * c2 + 1],
            )

        # dK retired -> Q(n) slot reusable.
        wgmma_dkv.wait_group[0]()
        warpgroup_fence(dk_acc)
        _ = empty[slot].arrive()

        sds_stage ^= 1
        q_base += BM
        slot += 2
        wrap += 1
        if wrap == 3:
            wrap = 0
            slot = 0
            phase ^= 1

    # ---- Epilogue. The dK/dV staging below overwrites the K/V smem
    # areas; warpgroup 0's final dQ GEMM (retired before its barrier
    # arrival) must not still be reading kt_view -> sync first.
    named_barrier[Int32(NWG * 128)](Int32(4))

    # dK *= scale; stage + TMA store dV then dK.
    var scale_acc: Scalar[accum_type] = softmax_scale.cast[accum_type]()
    comptime for c in range(c_frag_dkv):
        dk_acc.ptr[c] *= scale_acc

    var row_warp_base: Int = wg * WGMMA_M + warp_in_wg * 16

    # dV via the V smem area (dead), 16B-chunk-major for the
    # unswizzled store descriptor.
    comptime for c2 in range(c_frag_dkv // 2):
        comptime col_chunk: Int = c2 // 2
        comptime is_bot: Int = c2 % 2
        var row: Int = row_warp_base + lane_group + (8 if is_bot == 1 else 0)
        var pair = SIMD[dtype, 2](
            dv_acc.ptr[2 * c2].cast[dtype](),
            dv_acc.ptr[2 * c2 + 1].cast[dtype](),
        )
        (
            v_base + col_chunk * (BN * 8) + row * 8 + 2 * lane_pair
        ).store[width=2, alignment=4](pair)
    # dK via the K smem area (dead).
    comptime for c2 in range(c_frag_dkv // 2):
        comptime col_chunk: Int = c2 // 2
        comptime is_bot: Int = c2 % 2
        var row: Int = row_warp_base + lane_group + (8 if is_bot == 1 else 0)
        var pair = SIMD[dtype, 2](
            dk_acc.ptr[2 * c2].cast[dtype](),
            dk_acc.ptr[2 * c2 + 1].cast[dtype](),
        )
        (
            k_base + col_chunk * (BN * 8) + row * 8 + 2 * lane_pair
        ).store[width=2, alignment=4](pair)

    fence_async_view_proxy()
    named_barrier[Int32(NWG * 128)](Int32(4))
    if thread_idx.x == 128:
        var dv_st = LayoutTensor[
            dtype,
            Layout.row_major(BN, D),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](v_base)
        var dk_st = LayoutTensor[
            dtype,
            Layout.row_major(BN, D),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
            alignment=128,
        ](k_base)
        dv_tma.async_store_3d(dv_st, (0, h_idx, kv_row))
        dk_tma.async_store_3d(dk_st, (0, h_idx, kv_row))
        dv_tma.commit_group()
        dv_tma.wait_group()


# ===================================================================
# Preprocess kernel: dpsum = rowsum(dO * O), lse_log2 = lse*log2(e),
# zero dq_accum. Grid (S/128, H, B), 128 threads (one Q row each).
# ===================================================================
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kBwdPreThreads)
    )
)
def bwd_preprocess_kernel[
    dtype: DType,
    head_dim: Int,
](
    o_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    do_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dpsum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    lse_log2_ptr: UnsafePointer[Float32, MutAnyOrigin],
    dq_accum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int,
    nheads: Int,
):
    comptime D: Int = head_dim
    comptime BM: Int = kBwdPreBlockM
    comptime VEC: Int = 8

    var m_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var tid: Int = Int(thread_idx.x)

    # Coalesced: 8 threads per row, 16 rows per pass, 8 passes.
    # Each thread loads a contiguous 16-element (32B) slice.
    comptime LANES_PER_ROW: Int = 8
    comptime RVEC: Int = D // LANES_PER_ROW  # 16
    comptime ROWS_PER_PASS: Int = kBwdPreThreads // LANES_PER_ROW  # 16
    var sub: Int = tid % LANES_PER_ROW
    var row_in_pass: Int = tid // LANES_PER_ROW
    comptime for p in range(BM // ROWS_PER_PASS):
        var s: Int = m_block * BM + p * ROWS_PER_PASS + row_in_pass
        var off: Int = ((b_idx * seq_len + s) * nheads + h_idx) * D + sub * RVEC
        var o_v = (o_ptr + off).load[width=RVEC]().cast[DType.float32]()
        var do_v = (do_ptr + off).load[width=RVEC]().cast[DType.float32]()
        var part: Float32 = (o_v * do_v).reduce_add()
        var dps = warp.lane_group_sum[num_lanes=LANES_PER_ROW](part)
        if sub == 0:
            var bh_row: Int = (b_idx * nheads + h_idx) * seq_len + s
            (dpsum_ptr + bh_row)[0] = dps
            (lse_log2_ptr + bh_row)[0] = (lse_ptr + bh_row)[0] * Float32(
                log2e
            )

    # Cooperative coalesced zeroing of this block's dq_accum slice:
    # dq_accum is (B, H, D, S); this block owns s in [m*BM, m*BM+BM)
    # for every d. 4 d-rows per pass (32 lanes x 4 f32 = the 128 s).
    comptime ZVEC: Int = 4
    var dq_bh: Int = (b_idx * nheads + h_idx) * D
    var lane32: Int = tid % 32
    var dsub: Int = tid // 32
    comptime for pass_i in range(D // (kBwdPreThreads // 32)):
        var d: Int = pass_i * (kBwdPreThreads // 32) + dsub
        (
            dq_accum_ptr
            + (dq_bh + d) * seq_len
            + m_block * BM
            + lane32 * ZVEC
        ).store[width=ZVEC](SIMD[DType.float32, ZVEC](0))


# ===================================================================
# Convert kernel: dq = (dq_accum * softmax_scale) cast to bf16.
# Grid (S/128, H, B), 128 threads (one Q row each).
# ===================================================================
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(kBwdPreThreads)
    )
)
def bwd_convert_kernel[
    dtype: DType,
    head_dim: Int,
](
    dq_accum_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dq_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    seq_len: Int,
    nheads: Int,
    softmax_scale: Float32,
):
    """dq[b,s,h,d] = scale * dq_accum[b,h,d,s] via a 128x128 smem
    transpose tile (both gmem sides coalesced)."""
    comptime D: Int = head_dim
    comptime BM: Int = kBwdPreBlockM
    comptime PAD: Int = 4  # pad smem rows to dodge bank conflicts

    var m_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var tid: Int = Int(thread_idx.x)

    var tile = external_memory[
        Float32,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    # Phase 1: read (d, s-contiguous) rows of dq_accum, 4 d per pass.
    var dq_bh: Int = (b_idx * nheads + h_idx) * D
    var lane32: Int = tid % 32
    var dsub: Int = tid // 32
    comptime for pass_i in range(D // (kBwdPreThreads // 32)):
        var d: Int = pass_i * (kBwdPreThreads // 32) + dsub
        var v = (
            dq_accum_ptr
            + (dq_bh + d) * seq_len
            + m_block * BM
            + lane32 * 4
        ).load[width=4]()
        comptime for j in range(4):
            tile[d * (BM + PAD) + lane32 * 4 + j] = v[j]
    barrier()

    # Phase 2: thread t writes output row s = m*BM + t (contiguous
    # in d), reading the smem column s_local = t.
    comptime OV: Int = 8
    var s: Int = m_block * BM + tid
    var dq_off: Int = ((b_idx * seq_len + s) * nheads + h_idx) * D
    comptime for i in range(D // OV):
        var out = SIMD[dtype, OV]()
        comptime for j in range(OV):
            out[j] = (
                tile[(i * OV + j) * (BM + PAD) + tid] * softmax_scale
            ).cast[dtype]()
        (dq_ptr + dq_off + i * OV).store[width=OV](out)
