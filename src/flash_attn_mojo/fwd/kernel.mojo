"""Flash-attention forward kernel — port of modular's `mha_single_batch`.

Adapted from `modular/max/kernels/src/nn/attention/gpu/mha.mojo`
(function `mha_single_batch`, lines 1772-2480 at tag `max/v26.3.0`).
The original is a feature-rich kernel parameterised over `MHAOperand`
(KV-cache vs dense vs ragged), `MHAMask` (causal / materialised /
null), sink-weights, MQA/GQA `group`, varlen, etc. We strip everything
outside our envelope and call it directly with raw pointers + strides:

  - dtype fp16, head_dim = 64
  - non-causal, no dropout, no alibi, no softcap, no window
  - no MQA/GQA (kv_num_heads == num_heads)
  - dense (no kv-cache, no varlen)
  - one block per (q-tile, head, batch); 4 warps × 1 = 128 threads
  - tile config: BM=64, BN=64, BK=32, WM=16, WN=64

The arithmetic structure is identical to upstream's:
  1. Async-copy Q into smem (one BM×depth tile, never reloaded).
  2. For each KV tile of BN rows:
     a. Async-copy K into smem.
     b. `multistage_mma(p_reg, Q, K, transpose_b=True)`  → P scores.
     c. Apply scale·log2e to P (we use exp2 in the softmax).
     d. `_online_softmax_iter_for_mma_output` — per-row max/sum with
        warp reduce, applies the exp2-correction to the running
        output_reg_tile and stores new max/sum into rowmax/rowsum.
     e. Async-copy V into smem.
     f. `multistage_mma(output_reg, P, V, transpose_b=False)` —
        accumulates into output_reg_tile. With num_warps_n=1 P stays
        in registers; no _copy_frag_to_smem step.
  3. Normalise output_reg_tile by 1/rowsum.
  4. Stage output through smem (reusing q_smem buffer) and write to
     gmem with a swizzled vectorised copy.

Smem layout (dynamic, sized by `launch_fwd::shared_mem_bytes`):
    [ q_smem (BM × depth × fp16)              = 64 × 64 × 2 = 8 KiB ]
    [ k_smem (BN × depth × fp16)              = 64 × 64 × 2 = 8 KiB ]
    [ v_smem (BN × BN   × fp16)               = 64 × 64 × 2 = 8 KiB ]
    [ p_smem (BM × BN   × fp16) — only used when num_warps_n > 1     ]
The output write-back stage reuses q_smem in place (fp16 buffer,
same size). Total: ~24 KiB.
"""

from std.collections import OptionalReg
from std.math import log2, recip, exp, tanh
from std.math.constants import log2e
from std.sys import align_of, simd_width_of, size_of
from std.algorithm.functional import tile_and_unswitch, unswitch
import std.gpu.primitives.warp as warp
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
)
from std.gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from std.memory import stack_allocation
from std.utils.index import StaticTuple
from std.utils.numerics import min_or_neg_inf, get_accum_type

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
    ThreadScope,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size, get_mma_shape

from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from nn.softmax import _online_softmax_iter_for_mma_output

from common import kNThreads, kBlockM, kBlockN, kBlockK, kWM, kWN


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreads))
)
def fwd_kernel[
    dtype: DType,
    head_dim: Int,
    causal: Bool,
](
    seq_len: Int,
    actual_seq_len: Int,
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
    nheads_kv: Int,
    softcap: Float32,
    lse_ptr: UnsafePointer[Float32, MutAnyOrigin],
    lse_b_stride: Int,
    lse_h_stride: Int,
    window_left: Int,
    window_right: Int,
    alibi_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    alibi_b_stride: Int,
    alibi_h_stride: Int,
):
    comptime accum_type = get_accum_type[dtype]()
    comptime simd_size: Int = simd_width_of[dtype]()
    comptime num_pipeline_stages: Int = 2
    comptime k_group_size: Int = 1

    comptime BM: Int = kBlockM
    comptime BN: Int = kBlockN
    comptime BK: Int = kBlockK
    comptime WM: Int = kWM
    comptime WN: Int = kWN
    comptime num_threads: Int = kNThreads
    comptime num_warps_m: Int = BM // WM
    comptime num_warps_n: Int = BN // WN
    comptime depth: Int = head_dim

    comptime assert num_warps_m * num_warps_n == num_threads // WARP_SIZE, (
        "warp tile / num_threads mismatch"
    )
    comptime assert num_warps_n == 1, (
        "this port specialises on num_warps_n == 1 so we keep P in registers"
    )

    var tid: UInt32 = UInt32(thread_idx.x)
    var warp_id_v: UInt32 = warp.broadcast(tid // UInt32(WARP_SIZE))
    var lane: UInt32 = UInt32(lane_id())

    # With num_warps_n == 1: warp_x == 0 for every warp; warp_y = warp_id.
    var warp_y: UInt32 = warp_id_v
    var warp_x: UInt32 = 0

    var q_tile_idx: UInt32 = UInt32(block_idx.x)
    var head_idx: UInt32 = UInt32(block_idx.y)
    var batch: UInt32 = UInt32(block_idx.z)
    # MQA/GQA: Q-head i attends to K/V-head i // group_size, where
    # group_size = nheads_q // nheads_kv. When nheads_kv == nheads_q
    # this reduces to identity (group_size == 1).
    var group_size: Int = nheads // nheads_kv
    var kv_head_idx: Int = Int(head_idx) // group_size

    # ---- Dynamic smem layout: Q, K, V back-to-back.
    comptime alignment = align_of[SIMD[dtype, simd_size]]()
    comptime q_smem_size: Int = BM * depth
    comptime k_smem_size: Int = BN * depth
    comptime v_smem_size: Int = BN * BN  # BN * depth when depth == BN

    var q_smem = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        dtype,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    dtype,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=AddressSpace.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[dtype]]()
    comptime IteratorTypeK = LayoutTensorIter[
        dtype,
        Layout.row_major(BN, BK),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[dtype]]()
    comptime IteratorTypeV = LayoutTensorIter[
        dtype,
        Layout.row_major(BK, BN),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    # ---- MMA shape + per-warp register tiles.
    comptime mma_shape = get_mma_shape[dtype, accum_type]()
    comptime MMA_M: Int = mma_shape[0]
    comptime MMA_N: Int = mma_shape[1]
    comptime MMA_K: Int = mma_shape[2]
    comptime num_m_mmas: Int = WM // MMA_M
    comptime num_n_mmas: Int = WN // MMA_N

    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size: Int = frag_size[2]
    comptime p_frag_simdwidth: Int = p_frag_size // 2
    comptime p_frag_align = align_of[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation[stack_alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation[stack_alignment=p_frag_align]()
        .fill(0)
    )

    # ---- Per-row running max/sum (online softmax state).
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    # Use a large finite negative sentinel instead of -inf so that the
    # exp2(prev_max - new_max) correction in the online softmax is well-
    # defined when a row's score tile is fully masked (e.g. by a tight
    # sliding window): both old and new max land at this sentinel, the
    # correction becomes exp2(0) = 1, and output_reg_tile is preserved
    # (0 * 1 = 0). For any row with at least one finite valid score the
    # sentinel is far below the real scores (bf16 Q·K with the typical
    # 1/sqrt(64) scale stays in the |s| < ~100 range), so exp2(sentinel
    # - real) underflows to 0 and the result is identical to the -inf
    # case.
    var neg_sentinel: Scalar[accum_type] = Scalar[accum_type](-1.0e30)

    comptime for i in range(0, WM, 2):
        rowmax.store(i, SIMD[accum_type, 2](neg_sentinel))
        rowsum.store(i, SIMD[accum_type, 2](0))

    # `p_smem` is allocated unconditionally because `multistage_mma` for
    # the 2nd MMA expects an `a_smem_iter` in SHARED address space, even
    # when we route P through registers (num_warps_n == 1, swizzle_a=False).
    # `warp_scratch` reduces the per-row max/sum across N-dim warps; with
    # num_warps_n == 1 it's still allocated (size 0 ⇒ no-op stride).
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[dtype]]()
    comptime IteratorTypeP = LayoutTensorIter[
        dtype,
        Layout.row_major(BM, BK),
        _,
        address_space=AddressSpace.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space=AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # ---- Async-copy Q into smem (only once — held across the KV loop).
    comptime q_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(UNKNOWN_VALUE, 1)
    )
    var q_tile_num_rows: Int = min(
        Int(BM), seq_len - Int(q_tile_idx) * Int(BM)
    )
    var q_batch_head_off: Int = (
        Int(batch) * q_b_stride + Int(head_idx) * q_h_stride
    )
    var q_tile_row_off: Int = Int(q_tile_idx) * Int(BM) * q_l_stride
    var q_gmem_block = LayoutTensor[
        dtype,
        q_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        q_ptr + q_batch_head_off + q_tile_row_off,
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=DType.int32](
                q_tile_num_rows, depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=DType.int32](
                q_l_stride, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    comptime q_num_vecs: Int = BM * BK // simd_size
    # One-frag-per-thread layout for Q's (BM, BK) = (64, 32) tile (8-elt vecs):
    # (64 rows × 4 vec_cols) = 256 vec entries, 128 threads → 2 per thread.
    # Default (32, 4) layout has each thread writing two rows (r and r+32)
    # one col apart, which trips the same swizzled-async-copy bug as the
    # V (16, 8) layout (row r+32 lands at the wrong physical smem cell
    # for the last-row group → row 63 of Q comes out wrong). Pin to one
    # vec per thread along the row axis (64 × 2 thread layout).
    comptime async_copy_q_layout = Layout.row_major(
        BM,
        num_threads // BM,
    )

    comptime for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]
        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )
        q_gmem_iter._incr()

    var scale_log2e: Scalar[accum_type] = (
        softmax_scale.cast[accum_type]() * log2e
    )
    # Softcap (Gemma 2 / Grok): when > 0, replace scores s with
    # softcap * tanh(s * softmax_scale / softcap). The downstream
    # softmax uses exp2, so we still need to fold log2e in after the
    # tanh by scaling the post-tanh value by log2e (and `softcap`
    # itself absorbs into that constant). When softcap == 0 we keep
    # the fast `p_reg *= scale_log2e` path unchanged.
    var has_softcap: Bool = softcap != Float32(0)
    var softcap_acc: Scalar[accum_type] = softcap.cast[accum_type]()
    var softcap_inv_scaled: Scalar[accum_type] = (
        (softmax_scale / softcap).cast[accum_type]() if has_softcap
        else Scalar[accum_type](0)
    )
    var softcap_log2e: Scalar[accum_type] = softcap_acc * log2e

    # Sliding-window: `window_left == -1` means no left bound (= +inf
    # past), `window_right == -1` means no right bound. The fast path
    # `has_window == False` (both -1) keeps the original score-mask cost
    # at zero.
    var has_window: Bool = window_left != -1 or window_right != -1

    # ALiBi: `alibi_ptr` is null (addr==0) when the caller passes
    # `alibi_slopes=None` — the no-alibi fast path stays zero-overhead
    # behind this runtime gate. When non-null, load this block's slope
    # once at the top (one fp32 load per block) and bake `log2e` into
    # it so the per-element add lands in the log2 domain the rest of
    # the inner loop operates in.
    var has_alibi: Bool = (
        Int(alibi_ptr) != 0
    )
    var alibi_slope_log2e: Scalar[accum_type] = Scalar[accum_type](0)
    if has_alibi:
        var alibi_off: Int = (
            Int(batch) * alibi_b_stride + Int(head_idx) * alibi_h_stride
        )
        var slope_f32: Float32 = (alibi_ptr + alibi_off)[0]
        alibi_slope_log2e = slope_f32.cast[accum_type]() * log2e

    # ---- KV loop body (one (BN-tall) tile per iteration).
    @__copy_capture(
        seq_len,
        actual_seq_len,
        scale_log2e,
        has_softcap,
        softcap_inv_scaled,
        softcap_log2e,
        has_window,
        window_left,
        window_right,
        has_alibi,
        alibi_slope_log2e,
        neg_sentinel,
    )
    @always_inline
    @parameter
    def loop_over_kv[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth), IntTuple(UNKNOWN_VALUE, 1)
        )
        var kv_tile_num_rows: Int = min(tile_size, end - kv_tile_start_row)

        var k_base_off: Int = (
            Int(batch) * k_b_stride + kv_head_idx * k_h_stride
        )
        var k_row_off: Int = kv_tile_start_row * k_l_stride
        var k_runtime_layout = RuntimeLayout[
            kv_gmem_layout,
            element_type=DType.int32,
            linear_idx_type=DType.int32,
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                k_l_stride, 1
            ),
        )
        var k_gmem_block = LayoutTensor[
            dtype,
            kv_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](k_ptr + k_base_off + k_row_off, k_runtime_layout)
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_base_off: Int = (
            Int(batch) * v_b_stride + kv_head_idx * v_h_stride
        )
        var v_row_off: Int = kv_tile_start_row * v_l_stride
        var v_runtime_layout = RuntimeLayout[
            kv_gmem_layout,
            element_type=DType.int32,
            linear_idx_type=DType.int32,
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=DType.int32](
                v_l_stride, 1
            ),
        )
        var v_gmem_block = LayoutTensor[
            dtype,
            kv_gmem_layout,
            layout_int_type=DType.int32,
            linear_idx_type=DType.int32,
            masked=not not_last_iter,
        ](v_ptr + v_base_off + v_row_off, v_runtime_layout)
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q · Kᵀ — register-tile accumulator, zero each iter.
        _ = p_reg_tile.fill(0)

        comptime kv_num_vecs: Int = BN * BK // simd_size
        # Same swizzle-aware layout fix as Q: one vec entry per row per thread
        # to avoid the (32, 4) layout's row-31/row-63 swizzle aliasing.
        comptime async_copy_k_layout = Layout.row_major(
            BN,
            num_threads // BN,
        )

        comptime for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]
            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter[].vectorize[1, simd_size](),
            )
            k_gmem_iter._incr()

        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b for Q · Kᵀ
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=depth // BK,
            k_group_size=k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            depth // BK,
        )

        # Apply softmax_scale * log2e (we use exp2 inside online softmax).
        # Boundary-mask scores for keys >= actual_seq_len. Launcher pads
        # K/V/seq_len up to a multiple of BN, so `not_last_iter` (from
        # tile_and_unswitch over padded seq_len) is always True. Instead
        # we trigger masking whenever this tile overlaps the padding tail.
        var tile_needs_mask: Bool = (
            kv_tile_start_row + Int(BN) > actual_seq_len
        )
        # Causal mask only matters when the diagonal cuts through this tile.
        # A pair (q_row, kv_col) is above-diagonal iff kv_col > q_row. The
        # smallest q_row in this Q-tile is q_tile_idx*BM; the largest kv_col
        # in this tile is kv_tile_start_row + BN - 1. So the tile contains
        # at least one above-diagonal element iff
        #   kv_tile_start_row + BN - 1 > q_tile_idx*BM
        # i.e. kv_tile_start_row + BN > q_tile_idx*BM + 1. With BM == BN
        # and the block-skip cap at (q_tile_idx+1)*BM this is equivalent
        # to "this is the diagonal tile (kv_tile_start_row == q_tile_idx*BM)".
        var causal_tile_needs_mask: Bool = causal and (
            kv_tile_start_row + Int(BN)
            > Int(q_tile_idx) * Int(BM) + 1
        )
        # Per-lane row indices (m16n8k16 C-frag layout: i=0 → row lane/4,
        # i=1 → row lane/4 + 8 inside the WM×MMA_N MMA sub-tile).
        var lane_row0: UInt32 = lane // UInt32(4)
        var q_tile_row_base: Int = Int(q_tile_idx) * Int(BM) + Int(warp_y) * Int(WM)
        # Per-lane column offset inside the MMA_N tile: cols 2*(lane%4) and
        # 2*(lane%4)+1 (i.e. p_frag_simdwidth = 2 contiguous columns).
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()
        comptime for m_mma in range(num_m_mmas):
            comptime for n_mma in range(num_n_mmas):
                comptime mma_id = n_mma * num_m_mmas + m_mma
                var mma_col_base: UInt32 = (
                    UInt32(warp_x) * UInt32(WN) + UInt32(n_mma * MMA_N)
                )
                var col_off: UInt32 = (
                    lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                )

                comptime for i in range(2):
                    if has_softcap:
                        # scores' = softcap * tanh(scores * softmax_scale / softcap)
                        # then multiply by log2e (folded into softcap_log2e).
                        # tanh(x) = 1 - 2/(exp(2x)+1). softcap bounds the
                        # input so overflow at large |x| is not a concern.
                        # Cast through fp32 explicitly so exp's
                        # is_floating_point constraint is statically
                        # satisfied (accum_type's float-ness flows from a
                        # `def` and isn't comptime-proven on this path).
                        var s32 = p_reg_vec2[mma_id, i].cast[DType.float32]() * (
                            softcap_inv_scaled.cast[DType.float32]()
                        )
                        var e2 = exp(s32 + s32)
                        var one = Scalar[DType.float32](1)
                        var two = Scalar[DType.float32](2)
                        var t = one - two / (e2 + one)
                        p_reg_vec2[mma_id, i] = (
                            t * softcap_log2e.cast[DType.float32]()
                        ).cast[accum_type]()
                    else:
                        p_reg_vec2[mma_id, i] = p_reg_vec2[mma_id, i] * scale_log2e

                    # Per-row query index for this score frag slot.
                    var q_idx: Int = (
                        q_tile_row_base
                        + Int(m_mma) * Int(MMA_M)
                        + Int(lane_row0)
                        + (8 if i == 1 else 0)
                    )

                    # ALiBi: matches upstream FA2 `apply_alibi` exactly.
                    # Causal:    scores += slope * col_idx
                    #            (per-row constant -slope*q absorbed by
                    #            softmax, equivalent to slope*(k-q)).
                    # Non-causal: scores -= slope * |row - col|
                    #            (with seqlen_k == seqlen_q here).
                    # `alibi_slope_log2e` has `log2e` baked in so the add
                    # lands in the log2 domain the rest of the inner loop
                    # operates in. Runtime-gated behind `has_alibi` so the
                    # no-alibi fast path is zero-overhead.
                    if has_alibi:
                        var score_col_a: Int = (
                            kv_tile_start_row
                            + Int(mma_col_base + col_off)
                        )

                        @parameter
                        if causal:
                            comptime for j in range(p_frag_simdwidth):
                                var col_j: Int = score_col_a + j
                                p_reg_vec2[mma_id, i][j] = (
                                    p_reg_vec2[mma_id, i][j]
                                    + alibi_slope_log2e
                                    * Scalar[accum_type](col_j)
                                )
                        else:
                            comptime for j in range(p_frag_simdwidth):
                                var delta: Int = score_col_a + j - q_idx
                                if delta < 0:
                                    delta = -delta
                                p_reg_vec2[mma_id, i][j] = (
                                    p_reg_vec2[mma_id, i][j]
                                    - alibi_slope_log2e
                                    * Scalar[accum_type](delta)
                                )

                    # Per-element OOB mask: set score for any key >=
                    # actual_seq_len to -inf so its softmax weight is zero.
                    if tile_needs_mask:
                        var score_col: Int = (
                            kv_tile_start_row
                            + Int(mma_col_base + col_off)
                        )
                        var ne_inf = SIMD[accum_type, p_frag_simdwidth](
                            neg_sentinel
                        )

                        comptime for j in range(p_frag_simdwidth):
                            if score_col + j >= actual_seq_len:
                                p_reg_vec2[mma_id, i][j] = ne_inf[j]

                    # Causal mask: key column j must satisfy j <= q_idx.
                    if causal_tile_needs_mask:
                        var score_col_c: Int = (
                            kv_tile_start_row
                            + Int(mma_col_base + col_off)
                        )
                        var ne_inf_c = SIMD[accum_type, p_frag_simdwidth](
                            neg_sentinel
                        )

                        comptime for j in range(p_frag_simdwidth):
                            if score_col_c + j > q_idx:
                                p_reg_vec2[mma_id, i][j] = ne_inf_c[j]

                    # Sliding-window mask: -1 on either side means
                    # "unbounded on that side". The block-skip in the
                    # outer loop already removes whole tiles fully
                    # outside the window — this handles the partial-
                    # overlap boundary tiles.
                    if has_window:
                        var score_col_w: Int = (
                            kv_tile_start_row
                            + Int(mma_col_base + col_off)
                        )
                        var ne_inf_w = SIMD[accum_type, p_frag_simdwidth](
                            neg_sentinel
                        )

                        comptime for j in range(p_frag_simdwidth):
                            var kv_idx_w: Int = score_col_w + j
                            if (
                                window_left != -1
                                and kv_idx_w < q_idx - window_left
                            ):
                                p_reg_vec2[mma_id, i][j] = ne_inf_w[j]
                            if (
                                window_right != -1
                                and kv_idx_w > q_idx + window_right
                            ):
                                p_reg_vec2[mma_id, i][j] = ne_inf_w[j]

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )
        _online_softmax_iter_for_mma_output[
            accum_type,
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[2 * num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        # V's smem tile is (BK, BN). Use the same one-vec-per-row async
        # copy layout as Q and K (see comment above q's layout). With the
        # default (BN/simd_size, simd_size) = (16, 8) layout each thread
        # writes two destination fragments (rows ti and ti+16); the
        # stdlib's swizzled-async-copy formula then produces the wrong
        # physical address for the second fragment and V's row 31 ends up
        # holding V[16] data. Pin to one frag per thread (32 rows × 4
        # cols) to dodge the swizzle aliasing.
        comptime async_copy_v_layout = Layout.row_major(
            v_smem_iter.layout.shape[0].value(),
            num_threads // v_smem_iter.layout.shape[0].value(),
        )

        comptime for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]
            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle=v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_gmem_iter[].vectorize[1, simd_size](),
            )
            v_gmem_iter._incr()

        async_copy_commit_group()

        # num_warps_n == 1: keep P in regs as input to the 2nd MMA.
        # Reinterpret p_reg_tile's (num_m_mmas*num_n_mmas, p_frag_size)
        # layout as an iterator over (MMA_K/MMA_N * num_m_mmas, p_frag_size)
        # tiles — the inner "n_mmas" dim of the first MMA becomes the
        # "k_mmas" dim of the second.
        var p_reg_iter = p_reg_tile.tiled_iterator[
            MMA_K // MMA_N * num_m_mmas, p_frag_size
        ](0, 0)

        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            False,  # transpose_b for P · V
            swizzle_a=False,
            prefetch_init=False,
            static_num_iters=BN // BK,
            k_group_size=k_group_size,
        ](
            output_reg_tile,
            p_reg_iter,
            v_smem_iter,
            p_smem_iter,
            v_smem_iter,
            BN // BK,
        )

    # Block-skip for causal: KV tiles starting strictly above the last
    # query row of this Q-tile are entirely above the diagonal and
    # contribute nothing. Cap the loop at (q_tile_idx + 1) * BM.
    var kv_start: Int = 0
    var kv_end: Int = seq_len
    @parameter
    if causal:
        kv_end = min(seq_len, (Int(q_tile_idx) + 1) * Int(BM))
    # Block-skip for sliding-window: shrink [kv_start, kv_end) to the
    # range of BN-aligned tiles that can overlap the window for *any*
    # query row in this Q-tile. Aligned-down on the low side, aligned-
    # up on the high side, then clipped to actual_seq_len (and capped
    # against the existing causal cap above on the high side).
    if has_window:
        if window_left != -1:
            var lo_raw: Int = Int(q_tile_idx) * Int(BM) - window_left
            if lo_raw < 0:
                lo_raw = 0
            # Round down to BN multiple so tiles stay aligned.
            kv_start = (lo_raw // Int(BN)) * Int(BN)
        if window_right != -1:
            var hi_raw: Int = (
                (Int(q_tile_idx) + 1) * Int(BM) + window_right
            )
            if hi_raw > actual_seq_len:
                hi_raw = actual_seq_len
            # Round up to BN multiple, then clamp against kv_end (which
            # already encodes causal cap + padded seq_len).
            var hi_aligned: Int = (
                (hi_raw + Int(BN) - 1) // Int(BN)
            ) * Int(BN)
            if hi_aligned < kv_end:
                kv_end = hi_aligned
    if kv_start >= kv_end:
        # Fully-masked Q-tile (window puts every query past every key).
        # Fall through: rowsum stays 0, output gets the defensive 0
        # treatment below. Skip the KV loop entirely.
        pass
    else:
        tile_and_unswitch[loop_over_kv, [BN]](kv_start, kv_end)

    # ---- Write LSE (log-sum-exp, natural log) before rowsum is consumed.
    # With our exp2-based online softmax: rowsum is sum_j exp2(scaled_j - rowmax),
    # and rowmax is in the log2 domain (= softmax_scale * log2e * max_score).
    # The natural-log LSE is:
    #   lse = log(sum_j exp(scale * s_j)) = (rowmax + log2(rowsum)) / log2e
    # If rowsum == 0 (fully masked row) write -inf so downstream code sees
    # a clean "no valid keys" sentinel.
    # Only lane%4 == 0 owns the row (4 lanes share the same row index).
    var lse_lane_group: UInt32 = lane // 4
    var lse_lane_tid_in_grp: UInt32 = lane % 4
    if lse_lane_tid_in_grp == 0:
        var lse_batch_head_off: Int = (
            Int(batch) * lse_b_stride + Int(head_idx) * lse_h_stride
        )
        var inv_log2e_f32: Float32 = Float32(1) / Float32(log2e)
        comptime for m_mma in range(num_m_mmas):
            comptime for i in range(2):
                var rm = rowmax[2 * m_mma + i]
                var rs = rowsum[2 * m_mma + i]
                var lse_val: Float32
                if rs == Scalar[accum_type](0):
                    lse_val = min_or_neg_inf[DType.float32]()
                else:
                    var rm_f32 = rm.cast[DType.float32]()
                    var rs_f32 = rs.cast[DType.float32]()
                    lse_val = (rm_f32 + log2(rs_f32)) * inv_log2e_f32
                var row_in_warp: Int = (
                    Int(m_mma) * Int(MMA_M)
                    + Int(lse_lane_group)
                    + (8 if i == 1 else 0)
                )
                var q_row: Int = (
                    Int(q_tile_idx) * Int(BM)
                    + Int(warp_y) * Int(WM)
                    + row_in_warp
                )
                if q_row < actual_seq_len:
                    (lse_ptr + lse_batch_head_off + q_row)[0] = lse_val

    # ---- Normalise by 1/rowsum.
    # Defensive: a fully-masked row (no in-bounds, in-diagonal keys) has
    # rowsum == 0 → recip → inf. For seq_len_q == seq_len_k every query
    # has at least one valid key (itself), so this only fires on padded
    # tail rows, which the launcher discards. Still, swap 0 → 1 so the
    # output is a clean 0.
    comptime for m_mma in range(num_m_mmas):
        if rowsum[2 * m_mma] == Scalar[accum_type](0):
            rowsum[2 * m_mma] = Scalar[accum_type](1)
        if rowsum[2 * m_mma + 1] == Scalar[accum_type](0):
            rowsum[2 * m_mma + 1] = Scalar[accum_type](1)
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(num_n_mmas):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    # ---- Stage output through smem (reuse q_smem buffer) → gmem.
    comptime output_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(UNKNOWN_VALUE, 1)
    )
    var o_batch_head_off: Int = (
        Int(batch) * o_b_stride + Int(head_idx) * o_h_stride
    )
    var o_tile_row_off: Int = Int(q_tile_idx) * Int(BM) * o_l_stride
    var output_gmem_tile = LayoutTensor[
        dtype,
        output_gmem_layout,
        layout_int_type=DType.int32,
        linear_idx_type=DType.int32,
        masked=True,
    ](
        o_ptr + o_batch_head_off + o_tile_row_off,
        RuntimeLayout[element_type=DType.int32, linear_idx_type=DType.int32](
            RuntimeTuple[output_gmem_layout.shape, element_type=DType.int32](
                q_tile_num_rows, depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type=DType.int32](
                o_l_stride, 1
            ),
        ),
    )

    # Per-lane write of each MMA c-fragment slot into gmem at its
    # m16n8k16 C-fragment position:
    #   c[0..1] at (row=group,   col=2*tid + {0,1}) — n_mma sub-tile shifted
    #   c[2..3] at (row=group+8, col=2*tid + {0,1})
    # where groupID = lane/4 and threadID_in_group = lane%4. We index gmem
    # by (batch, query_row, head, col) so non-contiguous strides "just work".
    # This is deliberately *not* `copy_local_to_dram` + the modular swizzled
    # smem-staging: that helper drops the tail row of each warp when used
    # as written in mha_single_batch (rows 15/31/47/63 came out zero); the
    # by-hand store sidesteps it. TODO: fix and switch back once we have a
    # repro of the modular helper failing in isolation.
    var lane_group: UInt32 = lane // 4
    var lane_tid_in_grp: UInt32 = lane % 4
    var warp_row_base: Int = Int(warp_y) * WM
    var warp_col_base: Int = Int(warp_x) * WN

    comptime for n_mma in range(num_n_mmas):
        var col_off: Int = (
            warp_col_base
            + n_mma * MMA_N
            + Int(lane_tid_in_grp) * 2
        )
        var row0: Int = warp_row_base + Int(lane_group)
        var row1: Int = warp_row_base + Int(lane_group) + 8

        var c0 = output_reg_tile.ptr[n_mma * p_frag_size + 0].cast[dtype]()
        var c1 = output_reg_tile.ptr[n_mma * p_frag_size + 1].cast[dtype]()
        var c2 = output_reg_tile.ptr[n_mma * p_frag_size + 2].cast[dtype]()
        var c3 = output_reg_tile.ptr[n_mma * p_frag_size + 3].cast[dtype]()

        var base: Int = o_batch_head_off + o_tile_row_off
        if row0 < q_tile_num_rows:
            (o_ptr + base + row0 * o_l_stride + col_off)[0] = c0
            (o_ptr + base + row0 * o_l_stride + col_off + 1)[0] = c1
        if row1 < q_tile_num_rows:
            (o_ptr + base + row1 * o_l_stride + col_off)[0] = c2
            (o_ptr + base + row1 * o_l_stride + col_off + 1)[0] = c3
