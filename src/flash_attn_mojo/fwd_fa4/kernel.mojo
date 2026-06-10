"""FA4-target flash-attention forward kernel (sm_90a, Hopper).

v4: warp-specialized like FA4 — 1 producer warpgroup (WG0, thread 0
issues all TMA loads) + 2 MMA warpgroups, 384 threads. K/V tiles
live in a single 6-slot smem ring (slot(K_n) = 2n%6, slot(V_n) =
(2n+1)%6) guarded by full/empty mbarrier pairs:

    full[i].init(1)     -> flipped by TMA expect_bytes completion
    empty[i].init(256)  -> flipped when every MMA thread arrives

The MMA warpgroups run FA4's intra-warpgroup overlap schedule (from
`flash_attn/cute/flash_fwd_sm90.py::mma_one_n_block_intrawg_overlap`):

    wait full K(n+1); commit QK(n+1) -> s_reg        (no wait)
    wait full V(n);   commit PV(n):  p_reg x V -> o_reg
    wait_group(1)   # QK(n+1) retired -> arrive empty[K(n+1)]
    softmax(n+1)    # overlaps PV(n) on the tensor core
    wait_group(0)   # PV(n) retired   -> arrive empty[V(n)]
    pack P(n+1) bf16; rescale o_reg

No block-wide barriers in the loop (v3's main stall). Single S /
single P register buffer keeps the consumer register count near
FA4's 168/thread; producer deallocates to 24 regs via setmaxnreg.

The exp2 uses the scaled-domain trick: rowmax is kept premultiplied
by softmax_scale*log2(e) so P = exp2(fma(s, scale_log2, -m)).

Grid: (ceildiv(seqlen, BM), nheads, batch). Block: 384 threads.

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
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
)
from std.gpu.sync import named_barrier, named_barrier_arrive
from std.memory import stack_allocation

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from common import (
    kFa4NThreads,
    kFa4BlockM,
    kFa4BlockN,
    kFa4NMmaWarpgroups,
    kFa4KVStages,
)

comptime WGMMA_M: Int = 64
comptime WGMMA_K: Int = 16
comptime NUM_PRODUCER_REGS: Int = 24
comptime NUM_CONSUMER_REGS: Int = 240


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kFa4NThreads))
)
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
def fwd_fa4_kernel[
    dtype: DType,
    head_dim: Int,
    q_tile_shape: IndexList[3],
    q_desc_shape: IndexList[3],
    kv_tile_shape: IndexList[3],
    kv_desc_shape: IndexList[3],
    o_tile_shape: IndexList[3],
    o_desc_shape: IndexList[3],
](
    q_tma: TMATensorTile[dtype, 3, q_tile_shape, q_desc_shape],
    k_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    v_tma: TMATensorTile[dtype, 3, kv_tile_shape, kv_desc_shape],
    o_tma: TMATensorTile[dtype, 3, o_tile_shape, o_desc_shape],
    seq_len: Int,
    softmax_scale: Float32,
):
    comptime BM: Int = kFa4BlockM
    comptime BN: Int = kFa4BlockN
    comptime D: Int = head_dim
    comptime NWG: Int = kFa4NMmaWarpgroups
    comptime STAGES: Int = kFa4KVStages
    comptime accum_type: DType = DType.float32
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

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
    comptime kv_slot_size: Int = BN * D

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
    var kv_smem_base = smem_base + q_smem_size

    var mbar_q = stack_allocation[
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
        mbar_q[0].init()
        comptime for s in range(STAGES):
            full[s].init(1)
            empty[s].init(Int32(NWG * 128))
    barrier()

    var m_block: Int = Int(block_idx.x)
    var h_idx: Int = Int(block_idx.y)
    var b_idx: Int = Int(block_idx.z)
    var num_kv_blocks: Int = (seq_len + BN - 1) // BN

    var wgid: Int = Int(thread_idx.x) // 128

    if wgid == 0:
        # ================= producer =================
        warpgroup_reg_dealloc[NUM_PRODUCER_REGS]()
        if thread_idx.x == 0:
            mbar_q[0].expect_bytes(Int32(BM * D * size_of[dtype]()))
            q_tma.async_copy_3d(
                q_smem, mbar_q[0], (0, h_idx, b_idx * seq_len + m_block * BM)
            )
            # Incremental ring state: K(n) in slot 2n%6, V(n) in
            # (2n+1)%6; the empty-barrier phase flips every 3 tiles.
            var slot: Int = 0
            var phase: UInt32 = 0
            var wrap: Int = 0
            var row: Int = b_idx * seq_len
            for _ in range(num_kv_blocks):
                empty[slot].wait(phase)
                var k_st = LayoutTensor[
                    dtype,
                    k_smem_layout,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](kv_smem_base + slot * kv_slot_size)
                full[slot].expect_bytes(Int32(BN * D * size_of[dtype]()))
                k_tma.async_copy_3d(k_st, full[slot], (0, h_idx, row))

                empty[slot + 1].wait(phase)
                var v_st = LayoutTensor[
                    dtype,
                    v_smem_layout,
                    MutAnyOrigin,
                    address_space=AddressSpace.SHARED,
                    alignment=128,
                ](kv_smem_base + (slot + 1) * kv_slot_size)
                full[slot + 1].expect_bytes(Int32(BN * D * size_of[dtype]()))
                v_tma.async_copy_3d(v_st, full[slot + 1], (0, h_idx, row))

                row += BN
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

    # Unblock the producer's first ring cycle.
    comptime for s in range(STAGES):
        _ = empty[s].arrive()

    # Warp-scheduler pingpong (FA4's use_scheduler_barrier): named
    # barrier 1+wg gates each warpgroup's GEMM-issue phase; a
    # warpgroup arrives at the *other* one's barrier after committing
    # its GEMM pair, so issue phases alternate and each warpgroup's
    # softmax overlaps the other's GEMMs. WG0 self-arms its barrier.
    if wg == 0:
        named_barrier_arrive[Int32(NWG * 128)](Int32(1))

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

    # Online-softmax state, kept in the scaled (log2) domain:
    # rowmax_s = max(S) * softmax_scale * log2(e). 2 rows per thread.
    comptime rows_per_thread: Int = 2
    var rowmax = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var rowsum = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var scale_old = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    var neg_inf: Scalar[accum_type] = Scalar[accum_type](-1.0e30)
    comptime for i in range(rows_per_thread):
        rowmax[i] = neg_inf
        rowsum[i] = Scalar[accum_type](0)

    var scale_log2: Scalar[accum_type] = (
        softmax_scale * Scalar[DType.float32](log2e)
    ).cast[accum_type]()

    @parameter
    @always_inline
    def k_tile(slot: Int) -> LayoutTensor[
        dtype,
        k_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]:
        return {kv_smem_base + slot * kv_slot_size}

    @parameter
    @always_inline
    def v_tile(slot: Int) -> LayoutTensor[
        dtype,
        v_smem_layout,
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ]:
        return {kv_smem_base + slot * kv_slot_size}

    @parameter
    @always_inline
    def softmax_block():
        """Online softmax over s_reg (S just retired): update
        rowmax/rowsum/scale_old, write P (f32) back into s_reg."""
        var local_max = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        comptime for i in range(rows_per_thread):
            local_max[i] = neg_inf
        comptime for c in range(c_frag_size_qk):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            local_max[row_idx] = max(local_max[row_idx], s_reg.ptr[c])
        comptime for i in range(rows_per_thread):
            local_max[i] = warp.lane_group_max[num_lanes=4](local_max[i])
            var rmax_new: Scalar[accum_type] = max(
                local_max[i] * scale_log2, rowmax[i]
            )
            scale_old[i] = exp2(rowmax[i] - rmax_new)
            rowmax[i] = rmax_new

        var local_sum = stack_allocation[
            rows_per_thread, Scalar[accum_type]
        ]()
        comptime for i in range(rows_per_thread):
            local_sum[i] = Scalar[accum_type](0)
        comptime for c in range(c_frag_size_qk):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            var p: Scalar[accum_type] = exp2(
                s_reg.ptr[c].fma(scale_log2, -rowmax[row_idx])
            )
            s_reg.ptr[c] = p
            local_sum[row_idx] += p
        comptime for i in range(rows_per_thread):
            rowsum[i] = rowsum[i] * scale_old[i] + local_sum[i]

    @parameter
    @always_inline
    def pack_p():
        comptime for c in range(c_frag_size_qk):
            p_reg.ptr[c] = s_reg.ptr[c].cast[dtype]()

    @parameter
    @always_inline
    def rescale_o():
        comptime for c in range(c_frag_size_pv):
            comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
            o_reg.ptr[c] *= scale_old[row_idx]

    # ---- Prologue: S(0) -> P(0).
    mbar_q[0].wait(UInt32(0))
    full[0].wait(UInt32(0))
    warpgroup_fence(s_reg)
    wgmma_qk.arrive()
    wgmma_qk.wgmma[num_warp_groups=NWG, scale_c=0](
        q_smem, k_tile(0), s_reg, wg
    )
    wgmma_qk.commit_group()
    wgmma_qk.wait_group()
    warpgroup_fence(s_reg)
    _ = empty[0].arrive()

    softmax_block()  # rowmax starts at -inf -> scale_old==0, rowsum init
    pack_p()  # P(0)

    # ---- Main loop: QK(n+1) + PV(n) per iteration. Ring slots and
    # empty-barrier phases track incrementally (no div/mod per iter):
    # K(t): slot 2t%6, V(t): (2t+1)%6, phase flips every 3 tiles.
    var k_slot: Int = 2  # K(1)
    var k_phase: UInt32 = 0
    var k_wrap: Int = 1
    var v_slot: Int = 1  # V(0)
    var v_phase: UInt32 = 0
    var v_wrap: Int = 0

    for _ in range(num_kv_blocks - 1):
        # Queue QK(n+1) then PV(n) on the tensor core.
        full[k_slot].wait(k_phase)
        named_barrier[Int32(NWG * 128)](Int32(1 + wg))
        warpgroup_fence(s_reg)
        wgmma_qk.arrive()
        wgmma_qk.wgmma[num_warp_groups=NWG, scale_c=0](
            q_smem, k_tile(k_slot), s_reg, wg
        )
        wgmma_qk.commit_group()

        full[v_slot].wait(v_phase)
        warpgroup_fence(o_reg)
        wgmma_pv.arrive()
        wgmma_pv.wgmma(p_reg, v_tile(v_slot), o_reg)
        wgmma_pv.commit_group()
        named_barrier_arrive[Int32(NWG * 128)](Int32(2 - wg))

        # QK(n+1) retired (PV(n) still running on the tensor core).
        wgmma_qk.wait_group[1]()
        warpgroup_fence(s_reg)
        _ = empty[k_slot].arrive()

        # Softmax of S(n+1) overlaps PV(n).
        softmax_block()

        # PV(n) retired: p_reg and o_reg are safe to touch.
        wgmma_pv.wait_group[0]()
        warpgroup_fence(o_reg)
        _ = empty[v_slot].arrive()
        pack_p()  # P(n+1)
        rescale_o()

        k_slot += 2
        k_wrap += 1
        if k_wrap == 3:
            k_wrap = 0
            k_slot = 0
            k_phase ^= 1
        v_slot += 2
        v_wrap += 1
        if v_wrap == 3:
            v_wrap = 0
            v_slot = 1
            v_phase ^= 1

    # ---- Epilogue: PV(N-1) (v_slot/v_phase left at tile N-1).
    full[v_slot].wait(v_phase)
    warpgroup_fence(o_reg)
    wgmma_pv.arrive()
    wgmma_pv.wgmma(p_reg, v_tile(v_slot), o_reg)
    wgmma_pv.commit_group()
    wgmma_pv.wait_group[0]()
    warpgroup_fence(o_reg)

    # ---- Normalize (reciprocal; one div per row) and store.
    var inv_rowsum = stack_allocation[rows_per_thread, Scalar[accum_type]]()
    comptime for i in range(rows_per_thread):
        rowsum[i] = warp.lane_group_sum[num_lanes=4](rowsum[i])
        inv_rowsum[i] = Scalar[accum_type](1) / rowsum[i]

    comptime for c in range(c_frag_size_pv):
        comptime row_idx: Int = 1 if (c % 4) >= 2 else 0
        o_reg.ptr[c] *= inv_rowsum[row_idx]

    # ---- Store: stage O in smem (reusing the dead Q tile region)
    # and TMA bulk-store the whole tile. The unswizzled O descriptor
    # copies in 16B chunks along D (desc_shape = (BM, 1, 8)), chunk
    # j contiguous at offset j*BM*8: smem offset of (row, col) =
    # (col/8)*BM*8 + row*8 + col%8. A warp's 32-bit paired stores
    # land 512B-contiguous -> conflict-free. c-frag walk: col_chunk
    # = c/4 covers cols [8*col_chunk, 8*col_chunk+8) = exactly one
    # 16B chunk.
    var o_smem = LayoutTensor[
        dtype,
        Layout.row_major(BM, D),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](smem_base)

    var lane: Int = Int(lane_id())
    var warp_in_wg: Int = Int(warp_id()) % 4
    var lane_group: Int = lane // 4
    var lane_pair: Int = lane % 4
    var row_warp_base: Int = wg * WGMMA_M + warp_in_wg * 16

    comptime for c2 in range(c_frag_size_pv // 2):
        comptime col_chunk: Int = c2 // 2
        comptime is_bot: Int = c2 % 2
        var row: Int = row_warp_base + lane_group + (8 if is_bot == 1 else 0)
        var pair = SIMD[dtype, 2](
            o_reg.ptr[2 * c2].cast[dtype](),
            o_reg.ptr[2 * c2 + 1].cast[dtype](),
        )
        (
            o_smem.ptr + col_chunk * (BM * 8) + row * 8 + 2 * lane_pair
        ).store[width=2, alignment=4](pair)

    fence_async_view_proxy()
    # Producer warpgroup may have exited -> consumer-only barrier.
    # (id 3: ids 1-2 are the scheduler pingpong barriers.)
    named_barrier[Int32(NWG * 128)](Int32(3))
    if thread_idx.x == 128:
        o_tma.async_store_3d(
            o_smem, (0, h_idx, b_idx * seq_len + m_block * BM)
        )
        o_tma.commit_group()
        o_tma.wait_group()
