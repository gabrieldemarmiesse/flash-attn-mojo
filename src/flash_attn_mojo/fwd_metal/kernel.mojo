"""Apple-GPU (Metal) forward flash-attention kernel — the fast v1.

8x8 simdgroup-matrix MMA kernel for Apple silicon M1–M4 (apple9),
ccv/metal-flash-attention-shaped. Requires mojo >= 1.0.0b2 (the 8x8
`air.simdgroup_matrix` intrinsic SIGSEGVs the 1.0.0b1 compiler).

Algorithm (see METAL_PLAN.md for the full story): S = Q·K^T
accumulated in f16 via 8x8 MMAs, raw/unscaled like ccv; log2-domain
online softmax with the 2-hop xor(1)/xor(8) shuffle reductions (lanes
sharing a fragment row differ only in lane bits 0 and 3); P overwrites
S in place (the A-operand layout == the C layout, so the PV MMAs
consume it directly); O accumulated in f32 MMAs. All fragments are
read/written straight from device memory — no functional threadgroup
memory, zero barriers in the main loop (a dummy residency-pad
allocation stays as a throttle). Q is register-cached at D<=96 only
(ccv's table policy — caching at D=128 hits a register cliff);
otherwise streamed per traversal block straight from device.

Envelope: fp16 in / fp32 out, head_dim in {64, 128}, seqlen % 128 == 0.
"""

from std.gpu import barrier, block_idx, lane_id, thread_idx
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.math import exp2, log
from std.memory import stack_allocation
from std.sys import llvm_intrinsic

from common import BR, BC, LOG2E, RESIDENCY_PAD_ELEMS


@always_inline
def morton_order(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane element map (row, col_base).

    The lane owns elements (row, col_base) and (row, col_base + 1) of
    every 8x8 fragment. Lanes sharing a row differ only in lane bits 0
    and 3 — the row-wise softmax reductions are two shuffle-xor hops.
    """
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def mma8x8_f16(
    a: SIMD[DType.float16, 2],
    b: SIMD[DType.float16, 2],
    c: SIMD[DType.float16, 2],
) -> SIMD[DType.float16, 2]:
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float16, 2],
    ](a, b, c)


@always_inline
def mma8x8_f32(
    a: SIMD[DType.float16, 2],
    b: SIMD[DType.float16, 2],
    c: SIMD[DType.float32, 2],
) -> SIMD[DType.float32, 2]:
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, 2],
    ](a, b, c)


@always_inline
def _log2_f32(x: Float32) -> Float32:
    return log(x) * Float32(LOG2E)


def fwd_metal_kernel[
    D: Int
](
    q_g: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k_g: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_g: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    o_g: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    l_g: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    seq: Int,
    scale_log2: Float32,
):
    comptime dtype = DType.float16
    comptime ND = D // 8  # Q/O fragments per lane (D-dim tiles)
    comptime NC = BC // 8  # S/P fragments per lane (kv-dim tiles)
    comptime HB = 32  # head-block: D-chunk per Q reload when not cached
    # Q register-caching policy — mirrors ccv/MFA's table: cache at
    # D<=96 only (at D=128 the extra live fragments hit a register
    # cliff; the no-load skeleton runs 2.3x off roofline cached, at
    # roofline streamed).
    comptime Q_CACHED = D <= 96

    var residency_pad = stack_allocation[
        RESIDENCY_PAD_ELEMS,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    var lane = Int(lane_id())
    var sidx = Int(thread_idx.x) // 32
    var fl = morton_order(lane)
    var frow = fl[0]
    var fcol = fl[1]
    # 1-D grid, q-tile-major like ccv: consecutive threadgroup ids are
    # q tiles of the SAME head, so the ~concurrently-resident wave
    # shares one head's K/V stream (SLC hits).
    var n_q_tiles = seq // BR
    var q_tile = Int(block_idx.x) % n_q_tiles
    var head = Int(block_idx.x) // n_q_tiles

    var row = q_tile * BR + sidx * 8 + frow

    var q_head = q_g + head * seq * D
    var k_head = k_g + head * seq * D
    var v_head = v_g + head * seq * D

    var q_base = q_head + row * D + fcol
    var k_lane = k_head + frow
    var v_lane = v_head + fcol

    var q_frags = InlineArray[
        SIMD[dtype, 2], ND if Q_CACHED else 1
    ](uninitialized=True)
    comptime if Q_CACHED:
        comptime for di in range(ND):
            q_frags[di] = q_base.load[width=2](di * 8)

    var m = Float32(-3.0e38)
    var l = Float32(0)
    var o_frags = InlineArray[SIMD[DType.float32, 2], ND](
        fill=SIMD[DType.float32, 2](0)
    )

    for cb in range(seq // BC):
        var c = cb * BC

        # --- S = Q·K^T (raw, f16 accumulate; 8x8 MMAs) ---
        var s_frags = InlineArray[SIMD[dtype, 2], NC](
            fill=SIMD[dtype, 2](0)
        )
        var k_blk = k_lane + (c + fcol) * D
        comptime if Q_CACHED:
            comptime for di in range(ND):
                comptime for ci in range(NC):
                    var b = SIMD[dtype, 2](
                        k_blk.load(ci * 8 * D + di * 8),
                        k_blk.load(ci * 8 * D + di * 8 + D),
                    )
                    s_frags[ci] = mma8x8_f16(q_frags[di], b, s_frags[ci])
        else:
            for d_outer in range(0, D, HB):
                var qc = InlineArray[SIMD[dtype, 2], HB // 8](
                    uninitialized=True
                )
                comptime for di in range(HB // 8):
                    qc[di] = q_base.load[width=2](d_outer + di * 8)
                var k_chunk = k_blk + d_outer
                comptime for di in range(HB // 8):
                    comptime for ci in range(NC):
                        var b = SIMD[dtype, 2](
                            k_chunk.load(ci * 8 * D + di * 8),
                            k_chunk.load(ci * 8 * D + di * 8 + D),
                        )
                        s_frags[ci] = mma8x8_f16(qc[di], b, s_frags[ci])

        # --- online softmax (log2 domain, scale folded into exp2) ---
        var mx = s_frags[0]
        comptime for ci in range(1, NC):
            mx = max(mx, s_frags[ci])
        var m_tile = max(Float32(mx[0]), Float32(mx[1]))
        m_tile = max(m_tile, warp.shuffle_xor(m_tile, UInt32(1)))
        m_tile = max(m_tile, warp.shuffle_xor(m_tile, UInt32(8)))
        var m_new = m_tile * scale_log2

        var corr = Float32(1)
        if m_new > m:
            corr = exp2(m - m_new)
            m = m_new

        # P overwrites S in place; l sums the ROUNDED-to-dtype P (like
        # ccv) so O's normalization matches the PV accumulation.
        var l_pair = SIMD[DType.float32, 2](0)
        comptime for ci in range(NC):
            var p = exp2(
                s_frags[ci].cast[DType.float32]() * scale_log2 - m
            ).cast[dtype]()
            s_frags[ci] = p
            l_pair += p.cast[DType.float32]()
        var l_new = l_pair[0] + l_pair[1]
        l_new += warp.shuffle_xor(l_new, UInt32(1))
        l_new += warp.shuffle_xor(l_new, UInt32(8))
        l = l * corr + l_new

        # --- O = O*corr + P·V (f32 accumulate) ---
        comptime for di in range(ND):
            o_frags[di] = o_frags[di] * corr
        var v_blk = v_lane + (c + frow) * D
        comptime for ci in range(NC):
            comptime for di in range(ND):
                var vb = v_blk.load[width=2](ci * 8 * D + di * 8)
                o_frags[di] = mma8x8_f32(s_frags[ci], vb, o_frags[di])

    # Opaque use keeping the residency pad alive (seq is runtime; the
    # branch never executes).
    if seq < 0:
        residency_pad.store(lane, Scalar[dtype](l))
        barrier()
        l += Float32(residency_pad.load(0))

    # --- epilogue: O /= l, direct fragment stores; L = ln-domain LSE ---
    var inv_l = 1.0 / l
    comptime for di in range(ND):
        o_g.store(
            head * seq * D + row * D + di * 8 + fcol,
            o_frags[di] * inv_l,
        )
    if (lane & 9) == 0:  # one lane per fragment row (fcol == 0)
        l_g[head * seq + row] = (m + _log2_f32(l)) / Float32(LOG2E)
