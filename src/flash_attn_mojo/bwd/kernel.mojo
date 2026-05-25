"""Flash-attention backward kernel — tensor-core (multistage_mma) MMA path.

Replaces the scalar-loop MVP. Structure mirrors Tri Dao's FA2 bwd
(`flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h`) but uses
the same `multistage_mma` primitive the fwd kernel uses for both MMAs.

Algorithm (one block per (n_block, kv_head, batch)):

  1. Load K, V into smem.
  2. For each q-head in the MQA/GQA group, for each q-block:
     a. Load Q, dO into smem.
     b. Load LSE, delta.
     c. S  = Q · Kᵀ           (register-C, fp32, via multistage_mma)
     d. Softmax: p = exp(S*scale - LSE), apply softcap/alibi/mask/window/dropout.
     e. Write p_reg → PT_smem (transposed view) for use as A-operand of dV MMA.
     f. dP = dO · Vᵀ          (register-C, fp32)
     g. dS = p * (dP - delta[m]) * scale (with softcap chain rule).
     h. Write ds_reg → dST_smem (transposed) for use as A-operand of dK MMA.
     i. dV_acc += Pᵀ · dO      (A from PT_smem, B from dO_B)
     j. dQ_contrib = dS · K    (A from registers, B from K_B; atomic-add to dqaccum)
     k. dK_acc += dSᵀ · Q      (A from dST_smem, B from Q_B)
  3. Write dK_acc, dV_acc to gmem (cast to bf16).

Smem layout (BM=BN=64, D=64, BK=32):
    Q_A     (BM, D)  bf16  — (BM, BK) chunks, swizzled. A for S=Q·Kᵀ.
    Q_B     (BM, D)  bf16  — (BK, D) stripes, swizzled. B for dK=dSᵀ·Q.
    K_A     (BN, D)  bf16  — (BN, BK) chunks, swizzled. B for S=Q·Kᵀ.
    K_B     (BN, D)  bf16  — (BK, D) stripes, swizzled. B for dQ=dS·K.
    V       (BN, D)  bf16  — (BN, BK) chunks, swizzled. B for dP=dO·Vᵀ.
    dO_A    (BM, D)  bf16  — (BM, BK) chunks, swizzled. A for dP=dO·Vᵀ.
    dO_B    (BM, D)  bf16  — (BK, D) stripes, swizzled. B for dV=Pᵀ·dO.
    PT      (BN, BM) bf16  — (BN, BK) chunks, no swizzle. A for dV.
    dST     (BN, BM) bf16  — (BN, BK) chunks, no swizzle. A for dK.
    LSE, delta  (BM,) fp32 — one per Q row.

Total at D=64: 9 × BM*D*sizeof(bf16) + small ≈ 72.5 KiB.
At D=128:   ~144.5 KiB (fits H100's 228 KiB cap; over Ada's 99 KiB).
"""

from std.math import exp, log, log2, tanh
from std.math.constants import log2e
from std.sys import align_of, simd_width_of, size_of
from std.utils.index import StaticTuple
from std.utils.numerics import get_accum_type
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
)
import std.gpu.primitives.warp as warp
from std.gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from std.memory import stack_allocation
from std.atomic import Atomic

from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_dram_to_sram_async,
)
from layout.tensor_core import get_fragment_size, get_mma_shape

from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma

from common import kBwdNThreads, kBwdBlockM, kBwdBlockN


# BK chunk size along the K axis of every MMA. Picked to be a multiple
# of MMA_K (=16 for bf16 m16n8k16) and to divide BM, BN, and head_dim
# cleanly at the supported head_dims (32, 64, 128). Tried BK=64 at D≥64
# (fewer K-iters per MMA) and observed no measurable change in kernel
# time, so kept at 32 for uniformity with D=32.
comptime kBwdBlockK: Int = 32


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kBwdNThreads))
)
def bwd_kernel[
    dtype: DType,
    head_dim: Int,
    causal: Bool,
](
    seq_len: Int,
    nheads_q: Int,
    nheads_kv: Int,
    softmax_scale: Float32,
    softcap: Float32,
    dropout_p: Float32,
    rng_seed: UInt64,
    rng_offset: UInt64,
    q_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    do_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    delta_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    dk_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dqaccum_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alibi_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    alibi_b_stride: Int,
    alibi_h_stride: Int,
    window_left: Int,
    window_right: Int,
    q_b_stride: Int,
    q_l_stride: Int,
    q_h_stride: Int,
    k_b_stride: Int,
    k_l_stride: Int,
    k_h_stride: Int,
    v_b_stride: Int,
    v_l_stride: Int,
    v_h_stride: Int,
    do_b_stride: Int,
    do_l_stride: Int,
    do_h_stride: Int,
    dk_b_stride: Int,
    dk_l_stride: Int,
    dk_h_stride: Int,
    dv_b_stride: Int,
    dv_l_stride: Int,
    dv_h_stride: Int,
    lse_b_stride: Int,
    lse_h_stride: Int,
    delta_b_stride: Int,
    delta_h_stride: Int,
    dqa_b_stride: Int,
    dqa_h_stride: Int,
    dqa_l_stride: Int,
):
    # ---- Comptime configuration.
    comptime accum_type = get_accum_type[dtype]()
    comptime simd_size: Int = simd_width_of[dtype]()
    comptime alignment = align_of[SIMD[dtype, simd_size]]()
    comptime num_pipeline_stages: Int = 2
    comptime k_group_size: Int = 1

    comptime BM: Int = kBwdBlockM
    comptime BN: Int = kBwdBlockN
    comptime BK: Int = kBwdBlockK
    comptime D: Int = head_dim
    comptime num_threads: Int = kBwdNThreads

    # MMA shape (m16n8k16 for bf16/fp16, fp32 accum).
    comptime mma_shape = get_mma_shape[dtype, accum_type]()
    comptime MMA_M: Int = mma_shape[0]
    comptime MMA_N: Int = mma_shape[1]
    comptime MMA_K: Int = mma_shape[2]

    comptime WM: Int = BM // 4
    comptime WN_qk: Int = BN          # WN for S=Q·Kᵀ and dP=dO·Vᵀ
    comptime WN_dv: Int = D            # WN for dV / dQ / dK

    comptime num_m_mmas_qk: Int = WM // MMA_M
    comptime num_n_mmas_qk: Int = WN_qk // MMA_N
    comptime num_m_mmas_dv: Int = WM // MMA_M
    comptime num_n_mmas_dv: Int = WN_dv // MMA_N

    comptime num_iters_qk_k: Int = D // BK       # S, dP (K=D)
    comptime num_iters_bn_k: Int = BN // BK      # dQ (K=BN)
    comptime num_iters_bm_k: Int = BM // BK      # dV, dK (K=BM)

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime c_frag_size: Int = frag_size[2]
    comptime c_frag_simdwidth: Int = c_frag_size // 2
    comptime c_frag_align = align_of[SIMD[accum_type, c_frag_size]]()

    # ---- Per-thread / per-warp coordinates.
    var tid: Int = Int(thread_idx.x)
    var warp_id_v: UInt32 = warp.broadcast(UInt32(tid) // UInt32(WARP_SIZE))
    var warp_y: Int = Int(warp_id_v)
    var lane: Int = Int(lane_id())
    var lane_group: Int = lane // 4
    var lane_pair: Int = lane % 4

    var n_block: Int = Int(block_idx.x)
    var kv_head_idx: Int = Int(block_idx.y)
    var batch: Int = Int(block_idx.z)
    var group_size: Int = nheads_q // nheads_kv

    var kv_row_base: Int = n_block * BN
    if kv_row_base >= seq_len:
        return

    # ---- Dynamic smem carve-up.
    comptime buf_qd: Int = BM * D
    comptime buf_kv: Int = BN * D
    comptime buf_pt: Int = BN * BM

    var smem_base = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]()

    var q_a_smem = smem_base
    var q_b_smem = q_a_smem + buf_qd
    var k_a_smem = q_b_smem + buf_qd
    var k_b_smem = k_a_smem + buf_kv
    var v_smem   = k_b_smem + buf_kv
    var do_a_smem = v_smem + buf_kv
    var do_b_smem = do_a_smem + buf_qd
    var pt_smem  = do_b_smem + buf_qd
    var dst_smem = pt_smem + buf_pt
    var lse_smem = (dst_smem + buf_pt).bitcast[Float32]()
    var delta_smem = lse_smem + BM

    # ---- Smem iterators. Each is parameterized by its element layout
    # (one iter "step" worth) and walks the underlying buffer by
    # layout.size() elements per `_incr()`.
    comptime IterMK_BM = LayoutTensorIter[
        dtype,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
        circular=True,
    ]
    comptime IterMK_BN = LayoutTensorIter[
        dtype,
        Layout.row_major(BN, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
        circular=True,
    ]
    comptime IterKM = LayoutTensorIter[
        dtype,
        Layout.row_major(BK, D),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
        circular=True,
    ]

    var q_a_iter = IterMK_BM(q_a_smem, IterMK_BM.linear_uint_type(buf_qd))
    var q_b_iter = IterKM(q_b_smem, IterKM.linear_uint_type(buf_qd))
    var k_a_iter = IterMK_BN(k_a_smem, IterMK_BN.linear_uint_type(buf_kv))
    var k_b_iter = IterKM(k_b_smem, IterKM.linear_uint_type(buf_kv))
    var v_iter   = IterMK_BN(v_smem,    IterMK_BN.linear_uint_type(buf_kv))
    var do_a_iter = IterMK_BM(do_a_smem, IterMK_BM.linear_uint_type(buf_qd))
    var do_b_iter = IterKM(do_b_smem, IterKM.linear_uint_type(buf_qd))
    var pt_iter  = IterMK_BN(pt_smem,  IterMK_BN.linear_uint_type(buf_pt))
    var dst_iter = IterMK_BN(dst_smem, IterMK_BN.linear_uint_type(buf_pt))

    # ---- Register accumulators.
    var dv_acc = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas_dv * num_n_mmas_dv, c_frag_size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=c_frag_align]()
        .fill(0)
    )
    var dk_acc = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas_dv * num_n_mmas_dv, c_frag_size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=c_frag_align]()
        .fill(0)
    )

    var s_reg = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas_qk * num_n_mmas_qk, c_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=c_frag_align]()

    var dp_reg = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas_qk * num_n_mmas_qk, c_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=c_frag_align]()

    var dq_contrib = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas_dv * num_n_mmas_dv, c_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=c_frag_align]()

    # ---- Causal / window block-skip bounds.
    var num_q_blocks: Int = (seq_len + BM - 1) // BM
    var qb_start: Int = 0
    var qb_end: Int = num_q_blocks
    @parameter
    if causal:
        qb_start = (n_block * BN) // BM

    var has_window: Bool = window_left >= 0 or window_right >= 0
    if has_window:
        if window_right >= 0:
            var lo: Int = kv_row_base - window_right - BM + 1
            var qb_lo: Int = (lo + BM - 1) // BM if lo > 0 else 0
            if qb_lo > qb_start:
                qb_start = qb_lo
        if window_left >= 0:
            var hi_inclusive: Int = (
                kv_row_base + BN - 1 + window_left
            ) // BM
            var qb_hi: Int = hi_inclusive + 1
            if qb_hi < qb_end:
                qb_end = qb_hi

    # ---- Scale / softcap / dropout runtime state.
    # Cast Float32 inputs into accum_type once so the inner-loop math
    # stays in a single arithmetic domain (avoids spurious cast chains).
    var scale_a: Scalar[accum_type] = softmax_scale.cast[accum_type]()
    var softcap_a: Scalar[accum_type] = softcap.cast[accum_type]()
    var has_softcap: Bool = softcap > Float32(0)
    var softcap_inv_a: Scalar[accum_type] = (
        (Float32(1) / softcap).cast[accum_type]() if has_softcap
        else Scalar[accum_type](0)
    )

    var has_dropout: Bool = dropout_p > Float32(0)
    var keep_scale_a: Scalar[accum_type] = Scalar[accum_type](1)
    var log_keep_scale_a: Scalar[accum_type] = Scalar[accum_type](0)
    var drop_threshold_u32: UInt32 = UInt32(0)
    if has_dropout:
        keep_scale_a = (
            Float32(1) / (Float32(1) - dropout_p)
        ).cast[accum_type]()
        log_keep_scale_a = log(
            keep_scale_a.cast[DType.float32]()
        ).cast[accum_type]()
        var thr_f: Float32 = dropout_p * Float32(4294967296.0)
        if thr_f > Float32(4294967040.0):
            thr_f = Float32(4294967040.0)
        drop_threshold_u32 = UInt32(thr_f)
    var seed_mix: UInt64 = rng_seed ^ rng_offset
    var seed_mix_xor32: UInt32 = (
        UInt32(seed_mix & UInt64(0xFFFFFFFF)) ^ UInt32(seed_mix >> UInt64(32))
    )

    var has_alibi: Bool = Int(alibi_ptr) != 0

    var neg_sentinel: Scalar[accum_type] = Scalar[accum_type](-1.0e30)

    # ---- Load K (both layouts) and V into smem. Done once per block.
    var k_base_off: Int = batch * k_b_stride + kv_head_idx * k_h_stride
    var v_base_off: Int = batch * v_b_stride + kv_head_idx * v_h_stride
    var kv_actual_rows: Int = min(BN, seq_len - kv_row_base)
    if kv_actual_rows < 0:
        kv_actual_rows = 0

    # K → K_A (chunked) layout.
    comptime kv_gmem_layout = Layout(
        IntTuple(BN, D), IntTuple(UNKNOWN_VALUE, 1)
    )
    var k_gmem_block_a = LayoutTensor[
        dtype,
        kv_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        k_ptr + k_base_off + kv_row_base * k_l_stride,
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_actual_rows, D
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                k_l_stride, 1
            ),
        ),
    )
    var k_gmem_a_iter = k_gmem_block_a.tiled_iterator[BN, BK, axis=1](0, 0)
    comptime async_copy_bn = Layout.row_major(BN, num_threads // BN)
    comptime for k_id in range(D // BK):
        var smem_tile = k_a_iter.next_unsafe(
            k_a_iter.linear_uint_type(k_id)
        )[]
        copy_dram_to_sram_async[
            thread_layout=async_copy_bn,
            swizzle=True,
            num_threads=num_threads,
        ](
            smem_tile.vectorize[1, simd_size](),
            k_gmem_a_iter[].vectorize[1, simd_size](),
        )
        k_gmem_a_iter._incr()

    # K → K_B (striped) layout.
    var k_gmem_block_b = LayoutTensor[
        dtype,
        kv_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        k_ptr + k_base_off + kv_row_base * k_l_stride,
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_actual_rows, D
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                k_l_stride, 1
            ),
        ),
    )
    var k_gmem_b_iter = k_gmem_block_b.tiled_iterator[BK, D, axis=0](0, 0)
    comptime async_copy_bk = Layout.row_major(BK, num_threads // BK)
    comptime for m_id in range(BN // BK):
        var smem_tile = k_b_iter.next_unsafe(
            k_b_iter.linear_uint_type(m_id)
        )[]
        copy_dram_to_sram_async[
            thread_layout=async_copy_bk,
            swizzle=True,
            num_threads=num_threads,
        ](
            smem_tile.vectorize[1, simd_size](),
            k_gmem_b_iter[].vectorize[1, simd_size](),
        )
        k_gmem_b_iter._incr()

    # V → V_smem (chunked) layout.
    var v_gmem_block = LayoutTensor[
        dtype,
        kv_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        v_ptr + v_base_off + kv_row_base * v_l_stride,
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_actual_rows, D
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                v_l_stride, 1
            ),
        ),
    )
    var v_gmem_iter = v_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)
    comptime for k_id in range(D // BK):
        var smem_tile = v_iter.next_unsafe(
            v_iter.linear_uint_type(k_id)
        )[]
        copy_dram_to_sram_async[
            thread_layout=async_copy_bn,
            swizzle=True,
            num_threads=num_threads,
        ](
            smem_tile.vectorize[1, simd_size](),
            v_gmem_iter[].vectorize[1, simd_size](),
        )
        v_gmem_iter._incr()

    # ---- Per-q-head loop (MQA/GQA).
    for h_q_in_group in range(group_size):
        var q_head_idx: Int = kv_head_idx * group_size + h_q_in_group
        var q_base_off: Int = batch * q_b_stride + q_head_idx * q_h_stride
        var do_base_off: Int = batch * do_b_stride + q_head_idx * do_h_stride
        var lse_base_off: Int = (
            batch * lse_b_stride + q_head_idx * lse_h_stride
        )
        var delta_base_off: Int = (
            batch * delta_b_stride + q_head_idx * delta_h_stride
        )
        var dqa_base_off: Int = (
            batch * dqa_b_stride + q_head_idx * dqa_h_stride
        )

        var bh_mix: UInt64 = (
            UInt64(batch) * UInt64(2654435761)
            + UInt64(q_head_idx) * UInt64(40503)
        )
        var rng_key: UInt32 = (
            seed_mix_xor32
            ^ UInt32(bh_mix & UInt64(0xFFFFFFFF))
            ^ UInt32(bh_mix >> UInt64(32))
        )
        var alibi_slope_a: Scalar[accum_type] = Scalar[accum_type](0)
        if has_alibi:
            var alibi_off: Int = (
                batch * alibi_b_stride + q_head_idx * alibi_h_stride
            )
            alibi_slope_a = (alibi_ptr + alibi_off)[0].cast[accum_type]()

        for qb in range(qb_start, qb_end):
            var q_row_base: Int = qb * BM
            var q_actual_rows: Int = min(BM, seq_len - q_row_base)
            if q_actual_rows < 0:
                q_actual_rows = 0

            _ = s_reg.fill(0)
            _ = dp_reg.fill(0)
            _ = dq_contrib.fill(0)

            # ---- Load Q (both layouts).
            comptime qd_gmem_layout = Layout(
                IntTuple(BM, D), IntTuple(UNKNOWN_VALUE, 1)
            )
            var q_gmem_block_a = LayoutTensor[
                dtype,
                qd_gmem_layout,
                layout_int_type=DType.int32,
                linear_idx_type=DType.int32,
                masked=True,
            ](
                q_ptr + q_base_off + q_row_base * q_l_stride,
                RuntimeLayout[
                    element_type=DType.int32, linear_idx_type=DType.int32
                ](
                    RuntimeTuple[
                        qd_gmem_layout.shape, element_type=DType.int32
                    ](q_actual_rows, D),
                    RuntimeTuple[
                        qd_gmem_layout.stride, element_type=DType.int32
                    ](q_l_stride, 1),
                ),
            )
            var q_gmem_a_iter = q_gmem_block_a.tiled_iterator[BM, BK, axis=1](0, 0)
            comptime async_copy_bm = Layout.row_major(BM, num_threads // BM)
            comptime for k_id in range(D // BK):
                var smem_tile = q_a_iter.next_unsafe(
                    q_a_iter.linear_uint_type(k_id)
                )[]
                copy_dram_to_sram_async[
                    thread_layout=async_copy_bm,
                    swizzle=True,
                    num_threads=num_threads,
                ](
                    smem_tile.vectorize[1, simd_size](),
                    q_gmem_a_iter[].vectorize[1, simd_size](),
                )
                q_gmem_a_iter._incr()

            var q_gmem_block_b = LayoutTensor[
                dtype,
                qd_gmem_layout,
                layout_int_type=DType.int32,
                linear_idx_type=DType.int32,
                masked=True,
            ](
                q_ptr + q_base_off + q_row_base * q_l_stride,
                RuntimeLayout[
                    element_type=DType.int32, linear_idx_type=DType.int32
                ](
                    RuntimeTuple[
                        qd_gmem_layout.shape, element_type=DType.int32
                    ](q_actual_rows, D),
                    RuntimeTuple[
                        qd_gmem_layout.stride, element_type=DType.int32
                    ](q_l_stride, 1),
                ),
            )
            var q_gmem_b_iter = q_gmem_block_b.tiled_iterator[BK, D, axis=0](0, 0)
            comptime for m_id in range(BM // BK):
                var smem_tile = q_b_iter.next_unsafe(
                    q_b_iter.linear_uint_type(m_id)
                )[]
                copy_dram_to_sram_async[
                    thread_layout=async_copy_bk,
                    swizzle=True,
                    num_threads=num_threads,
                ](
                    smem_tile.vectorize[1, simd_size](),
                    q_gmem_b_iter[].vectorize[1, simd_size](),
                )
                q_gmem_b_iter._incr()

            # ---- Load dO (both layouts).
            var do_gmem_block_a = LayoutTensor[
                dtype,
                qd_gmem_layout,
                layout_int_type=DType.int32,
                linear_idx_type=DType.int32,
                masked=True,
            ](
                do_ptr + do_base_off + q_row_base * do_l_stride,
                RuntimeLayout[
                    element_type=DType.int32, linear_idx_type=DType.int32
                ](
                    RuntimeTuple[
                        qd_gmem_layout.shape, element_type=DType.int32
                    ](q_actual_rows, D),
                    RuntimeTuple[
                        qd_gmem_layout.stride, element_type=DType.int32
                    ](do_l_stride, 1),
                ),
            )
            var do_gmem_a_iter = do_gmem_block_a.tiled_iterator[BM, BK, axis=1](0, 0)
            comptime for k_id in range(D // BK):
                var smem_tile = do_a_iter.next_unsafe(
                    do_a_iter.linear_uint_type(k_id)
                )[]
                copy_dram_to_sram_async[
                    thread_layout=async_copy_bm,
                    swizzle=True,
                    num_threads=num_threads,
                ](
                    smem_tile.vectorize[1, simd_size](),
                    do_gmem_a_iter[].vectorize[1, simd_size](),
                )
                do_gmem_a_iter._incr()

            var do_gmem_block_b = LayoutTensor[
                dtype,
                qd_gmem_layout,
                layout_int_type=DType.int32,
                linear_idx_type=DType.int32,
                masked=True,
            ](
                do_ptr + do_base_off + q_row_base * do_l_stride,
                RuntimeLayout[
                    element_type=DType.int32, linear_idx_type=DType.int32
                ](
                    RuntimeTuple[
                        qd_gmem_layout.shape, element_type=DType.int32
                    ](q_actual_rows, D),
                    RuntimeTuple[
                        qd_gmem_layout.stride, element_type=DType.int32
                    ](do_l_stride, 1),
                ),
            )
            var do_gmem_b_iter = do_gmem_block_b.tiled_iterator[BK, D, axis=0](0, 0)
            comptime for m_id in range(BM // BK):
                var smem_tile = do_b_iter.next_unsafe(
                    do_b_iter.linear_uint_type(m_id)
                )[]
                copy_dram_to_sram_async[
                    thread_layout=async_copy_bk,
                    swizzle=True,
                    num_threads=num_threads,
                ](
                    smem_tile.vectorize[1, simd_size](),
                    do_gmem_b_iter[].vectorize[1, simd_size](),
                )
                do_gmem_b_iter._incr()

            # ---- Load LSE, delta.
            comptime LSE_PER_THREAD: Int = (BM + num_threads - 1) // num_threads
            comptime for li in range(LSE_PER_THREAD):
                var i: Int = li * num_threads + tid
                if i < BM:
                    var g_row: Int = q_row_base + i
                    if g_row < seq_len:
                        lse_smem[i] = (lse_ptr + lse_base_off + g_row)[0]
                        delta_smem[i] = (delta_ptr + delta_base_off + g_row)[0]
                    else:
                        lse_smem[i] = Float32(-1.0e30)
                        delta_smem[i] = Float32(0)

            async_copy_commit_group()
            async_copy_wait_all()
            barrier()

            # ---- MMA 1: s_reg = Q · Kᵀ.
            multistage_mma[
                BM, BN, BK, WM, WN_qk,
                num_threads, num_pipeline_stages,
                True,
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=num_iters_qk_k,
                k_group_size=k_group_size,
            ](
                s_reg,
                q_a_iter, k_a_iter,
                q_a_iter, k_a_iter,
                num_iters_qk_k,
            )

            # ---- Softmax → p_reg (overwrites s_reg).
            var s_vec = s_reg.vectorize[1, c_frag_simdwidth]()
            comptime for m_mma in range(num_m_mmas_qk):
                comptime for n_mma in range(num_n_mmas_qk):
                    comptime mma_id = n_mma * num_m_mmas_qk + m_mma
                    var mma_col_base: Int = n_mma * MMA_N
                    comptime for i in range(2):
                        var row_in_warp: Int = (
                            m_mma * MMA_M
                            + lane_group
                            + (8 if i == 1 else 0)
                        )
                        var lse_row: Int = warp_y * WM + row_in_warp
                        var q_idx: Int = q_row_base + lse_row
                        var lse_val: Scalar[accum_type] = (
                            lse_smem[lse_row].cast[accum_type]()
                        )
                        comptime for j in range(c_frag_simdwidth):
                            var col_off: Int = 2 * lane_pair + j
                            var g_col: Int = (
                                kv_row_base + mma_col_base + col_off
                            )
                            var s_pre: Scalar[accum_type] = (
                                s_vec[mma_id, i][j] * scale_a
                            )
                            var s_post: Scalar[accum_type] = s_pre
                            if has_softcap:
                                s_post = softcap_a * tanh(
                                    (s_pre * softcap_inv_a).cast[DType.float32]()
                                ).cast[accum_type]()
                            if has_alibi:
                                @parameter
                                if causal:
                                    s_post = (
                                        s_post
                                        + alibi_slope_a
                                        * Scalar[accum_type](g_col)
                                    )
                                else:
                                    var d_rc: Int = g_col - q_idx
                                    if d_rc < 0:
                                        d_rc = -d_rc
                                    s_post = (
                                        s_post
                                        - alibi_slope_a
                                        * Scalar[accum_type](d_rc)
                                    )
                            var masked: Bool = (
                                q_idx >= seq_len or g_col >= seq_len
                            )
                            @parameter
                            if causal:
                                if q_idx < g_col:
                                    masked = True
                            if has_window:
                                if (
                                    window_left >= 0
                                    and g_col < q_idx - window_left
                                ):
                                    masked = True
                                if (
                                    window_right >= 0
                                    and g_col > q_idx + window_right
                                ):
                                    masked = True
                            var p_val: Scalar[accum_type]
                            if masked:
                                p_val = Scalar[accum_type](0)
                            else:
                                p_val = exp(
                                    (s_post - lse_val).cast[DType.float32]()
                                ).cast[accum_type]()
                            if has_dropout and not masked:
                                var u: UInt32 = (
                                    rng_key
                                    ^ UInt32(q_idx) * UInt32(0x9E3779B1)
                                    ^ UInt32(g_col) * UInt32(0x85EBCA77)
                                )
                                u = u ^ (u >> UInt32(16))
                                u = u * UInt32(0x7FEB352D)
                                u = u ^ (u >> UInt32(15))
                                u = u * UInt32(0x846CA68B)
                                u = u ^ (u >> UInt32(16))
                                if u < drop_threshold_u32:
                                    p_val = Scalar[accum_type](0)
                                else:
                                    p_val = p_val * keep_scale_a
                            s_vec[mma_id, i][j] = p_val

            # ---- Write p_reg → PT_smem in transposed layout.
            comptime for m_mma in range(num_m_mmas_qk):
                comptime for n_mma in range(num_n_mmas_qk):
                    comptime mma_id_p = n_mma * num_m_mmas_qk + m_mma
                    var mma_col_base_p: Int = n_mma * MMA_N
                    var row_top: Int = (
                        warp_y * WM + m_mma * MMA_M + lane_group
                    )
                    var row_bot: Int = row_top + 8
                    var col_a: Int = mma_col_base_p + 2 * lane_pair
                    var col_b: Int = col_a + 1
                    var top_chunk: Int = row_top // BK
                    var top_off: Int = row_top - top_chunk * BK
                    var bot_chunk: Int = row_bot // BK
                    var bot_off: Int = row_bot - bot_chunk * BK
                    pt_smem[top_chunk * BN * BK + col_a * BK + top_off] = (
                        s_reg.ptr[mma_id_p * c_frag_size + 0].cast[dtype]()
                    )
                    pt_smem[top_chunk * BN * BK + col_b * BK + top_off] = (
                        s_reg.ptr[mma_id_p * c_frag_size + 1].cast[dtype]()
                    )
                    pt_smem[bot_chunk * BN * BK + col_a * BK + bot_off] = (
                        s_reg.ptr[mma_id_p * c_frag_size + 2].cast[dtype]()
                    )
                    pt_smem[bot_chunk * BN * BK + col_b * BK + bot_off] = (
                        s_reg.ptr[mma_id_p * c_frag_size + 3].cast[dtype]()
                    )

            # ---- MMA 3: dp_reg = dO · Vᵀ.
            _ = dp_reg.fill(0)
            multistage_mma[
                BM, BN, BK, WM, WN_qk,
                num_threads, num_pipeline_stages,
                True,
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=num_iters_qk_k,
                k_group_size=k_group_size,
            ](
                dp_reg,
                do_a_iter, v_iter,
                do_a_iter, v_iter,
                num_iters_qk_k,
            )

            # ---- Combine: ds_reg = p_reg * (dp_reg - delta[m]) * scale.
            var dp_vec = dp_reg.vectorize[1, c_frag_simdwidth]()
            comptime for m_mma in range(num_m_mmas_qk):
                comptime for n_mma in range(num_n_mmas_qk):
                    comptime mma_id_ds = n_mma * num_m_mmas_qk + m_mma
                    var mma_col_base_ds: Int = n_mma * MMA_N
                    comptime for i in range(2):
                        var row_in_warp: Int = (
                            m_mma * MMA_M
                            + lane_group
                            + (8 if i == 1 else 0)
                        )
                        var lse_row: Int = warp_y * WM + row_in_warp
                        var q_idx_ds: Int = q_row_base + lse_row
                        var delta_v: Scalar[accum_type] = (
                            delta_smem[lse_row].cast[accum_type]()
                        )
                        var lse_v: Scalar[accum_type] = (
                            lse_smem[lse_row].cast[accum_type]()
                        )
                        comptime for j in range(c_frag_simdwidth):
                            var col_off_ds: Int = 2 * lane_pair + j
                            var g_col_ds: Int = (
                                kv_row_base + mma_col_base_ds + col_off_ds
                            )
                            var p_v: Scalar[accum_type] = (
                                s_vec[mma_id_ds, i][j]
                            )
                            var ds: Scalar[accum_type] = (
                                p_v * (dp_vec[mma_id_ds, i][j] - delta_v)
                            )
                            if has_softcap:
                                if p_v > Scalar[accum_type](0):
                                    var s_post_r: Scalar[accum_type] = (
                                        log(
                                            p_v.cast[DType.float32]()
                                        ).cast[accum_type]()
                                        - log_keep_scale_a + lse_v
                                    )
                                    if has_alibi:
                                        @parameter
                                        if causal:
                                            s_post_r = (
                                                s_post_r
                                                - alibi_slope_a
                                                * Scalar[accum_type](g_col_ds)
                                            )
                                        else:
                                            var d_rc_r: Int = (
                                                g_col_ds - q_idx_ds
                                            )
                                            if d_rc_r < 0:
                                                d_rc_r = -d_rc_r
                                            s_post_r = (
                                                s_post_r
                                                + alibi_slope_a
                                                * Scalar[accum_type](d_rc_r)
                                            )
                                    var t_r: Scalar[accum_type] = (
                                        s_post_r * softcap_inv_a
                                    )
                                    ds = ds * (Scalar[accum_type](1) - t_r * t_r)
                            ds = ds * scale_a
                            s_vec[mma_id_ds, i][j] = ds

            # ---- Write ds_reg → dST_smem (same transposed layout as PT).
            comptime for m_mma in range(num_m_mmas_qk):
                comptime for n_mma in range(num_n_mmas_qk):
                    comptime mma_id_dst = n_mma * num_m_mmas_qk + m_mma
                    var mma_col_base_dst: Int = n_mma * MMA_N
                    var row_top: Int = (
                        warp_y * WM + m_mma * MMA_M + lane_group
                    )
                    var row_bot: Int = row_top + 8
                    var col_a: Int = mma_col_base_dst + 2 * lane_pair
                    var col_b: Int = col_a + 1
                    var top_chunk: Int = row_top // BK
                    var top_off: Int = row_top - top_chunk * BK
                    var bot_chunk: Int = row_bot // BK
                    var bot_off: Int = row_bot - bot_chunk * BK
                    dst_smem[top_chunk * BN * BK + col_a * BK + top_off] = (
                        s_reg.ptr[mma_id_dst * c_frag_size + 0].cast[dtype]()
                    )
                    dst_smem[top_chunk * BN * BK + col_b * BK + top_off] = (
                        s_reg.ptr[mma_id_dst * c_frag_size + 1].cast[dtype]()
                    )
                    dst_smem[bot_chunk * BN * BK + col_a * BK + bot_off] = (
                        s_reg.ptr[mma_id_dst * c_frag_size + 2].cast[dtype]()
                    )
                    dst_smem[bot_chunk * BN * BK + col_b * BK + bot_off] = (
                        s_reg.ptr[mma_id_dst * c_frag_size + 3].cast[dtype]()
                    )

            barrier()

            # ---- MMA 2: dV_acc += Pᵀ · dO.
            multistage_mma[
                BN, D, BK, WM, WN_dv,
                num_threads, num_pipeline_stages,
                False,
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters=num_iters_bm_k,
                k_group_size=k_group_size,
            ](
                dv_acc,
                pt_iter, do_b_iter,
                pt_iter, do_b_iter,
                num_iters_bm_k,
            )

            # ---- MMA 4: dQ_contrib = dS · K  (A from registers).
            var ds_reg_iter = s_reg.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas_qk, c_frag_size
            ](0, 0)
            multistage_mma[
                BM, D, BK, WM, WN_dv,
                num_threads, num_pipeline_stages,
                False,
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters=num_iters_bn_k,
                k_group_size=k_group_size,
            ](
                dq_contrib,
                ds_reg_iter, k_b_iter,
                pt_iter, k_b_iter,  # a_smem_iter placeholder (unused; A is LOCAL)
                num_iters_bn_k,
            )

            # ---- Atomic-add dq_contrib c-frag into gmem dqaccum.
            comptime for n_mma in range(num_n_mmas_dv):
                var col_base_dq: Int = n_mma * MMA_N + 2 * lane_pair
                var row_top: Int = warp_y * WM + lane_group
                var row_bot: Int = row_top + 8
                var g_row_top: Int = q_row_base + row_top
                var g_row_bot: Int = q_row_base + row_bot
                var c0_dq = dq_contrib.ptr[n_mma * c_frag_size + 0]
                var c1_dq = dq_contrib.ptr[n_mma * c_frag_size + 1]
                var c2_dq = dq_contrib.ptr[n_mma * c_frag_size + 2]
                var c3_dq = dq_contrib.ptr[n_mma * c_frag_size + 3]
                var base: Int = dqa_base_off
                if g_row_top < seq_len:
                    _ = Atomic.fetch_add(
                        dqaccum_ptr + base + g_row_top * dqa_l_stride + col_base_dq,
                        c0_dq.cast[DType.float32](),
                    )
                    _ = Atomic.fetch_add(
                        dqaccum_ptr + base + g_row_top * dqa_l_stride + col_base_dq + 1,
                        c1_dq.cast[DType.float32](),
                    )
                if g_row_bot < seq_len:
                    _ = Atomic.fetch_add(
                        dqaccum_ptr + base + g_row_bot * dqa_l_stride + col_base_dq,
                        c2_dq.cast[DType.float32](),
                    )
                    _ = Atomic.fetch_add(
                        dqaccum_ptr + base + g_row_bot * dqa_l_stride + col_base_dq + 1,
                        c3_dq.cast[DType.float32](),
                    )

            # ---- MMA 5: dK_acc += dSᵀ · Q.
            multistage_mma[
                BN, D, BK, WM, WN_dv,
                num_threads, num_pipeline_stages,
                False,
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters=num_iters_bm_k,
                k_group_size=k_group_size,
            ](
                dk_acc,
                dst_iter, q_b_iter,
                dst_iter, q_b_iter,
                num_iters_bm_k,
            )

            barrier()  # ensure PT/dST smem is free for next qb's writes

    # ---- Write dK_acc, dV_acc to gmem.
    var dk_base_off: Int = batch * dk_b_stride + kv_head_idx * dk_h_stride
    var dv_base_off: Int = batch * dv_b_stride + kv_head_idx * dv_h_stride
    comptime for n_mma in range(num_n_mmas_dv):
        var col_base_kv: Int = n_mma * MMA_N + 2 * lane_pair
        var row_top: Int = warp_y * WM + lane_group
        var row_bot: Int = row_top + 8
        var g_row_top: Int = kv_row_base + row_top
        var g_row_bot: Int = kv_row_base + row_bot
        var c0_dk = dk_acc.ptr[n_mma * c_frag_size + 0].cast[dtype]()
        var c1_dk = dk_acc.ptr[n_mma * c_frag_size + 1].cast[dtype]()
        var c2_dk = dk_acc.ptr[n_mma * c_frag_size + 2].cast[dtype]()
        var c3_dk = dk_acc.ptr[n_mma * c_frag_size + 3].cast[dtype]()
        var c0_dv = dv_acc.ptr[n_mma * c_frag_size + 0].cast[dtype]()
        var c1_dv = dv_acc.ptr[n_mma * c_frag_size + 1].cast[dtype]()
        var c2_dv = dv_acc.ptr[n_mma * c_frag_size + 2].cast[dtype]()
        var c3_dv = dv_acc.ptr[n_mma * c_frag_size + 3].cast[dtype]()
        if g_row_top < seq_len:
            (dk_ptr + dk_base_off + g_row_top * dk_l_stride + col_base_kv)[0] = c0_dk
            (dk_ptr + dk_base_off + g_row_top * dk_l_stride + col_base_kv + 1)[0] = c1_dk
            (dv_ptr + dv_base_off + g_row_top * dv_l_stride + col_base_kv)[0] = c0_dv
            (dv_ptr + dv_base_off + g_row_top * dv_l_stride + col_base_kv + 1)[0] = c1_dv
        if g_row_bot < seq_len:
            (dk_ptr + dk_base_off + g_row_bot * dk_l_stride + col_base_kv)[0] = c2_dk
            (dk_ptr + dk_base_off + g_row_bot * dk_l_stride + col_base_kv + 1)[0] = c3_dk
            (dv_ptr + dv_base_off + g_row_bot * dv_l_stride + col_base_kv)[0] = c2_dv
            (dv_ptr + dv_base_off + g_row_bot * dv_l_stride + col_base_kv + 1)[0] = c3_dv
