# bench_mojo_metal: v1 flash-attention FORWARD on Apple-silicon GPUs
# (Q,K,V only — non-causal, dense, fp16 in / fp32 out, MHA batched via
# grid.y like ccv's kernel).
#
# Run from the repo root:
#   .venv/bin/mojo run bench/bench_mojo_metal.mojo -- \
#       --seq 4096 --head-dim 128 --heads 16 --iters 10 [--check]
#
# v1 is the MMA kernel (requires mojo >= 1.0.0b2: the 8x8
# air.simdgroup_matrix intrinsic SIGSEGVs the 1.0.0b1 compiler). It is
# ccv/MFA-shaped — same engine, same blocking (parallelization=16,
# traversal=128), 2 simdgroups x 32 lanes, each simdgroup owning 8 q
# rows: S = Q·K^T accumulated in f16 via 8x8 simdgroup-matrix MMAs,
# raw (unscaled) like ccv; log2-domain online softmax with the
# 2-hop shuffle reduction (xor 1, xor 8 — lanes sharing a fragment
# row differ only in bits 0 and 3 of the lane id); P kept f16 in the
# S fragment layout (A-operand layout == C layout, so no conversion
# shuffle); O accumulated in f32 MMAs.
#
# Deliberate departures from the generated ccv kernel (fast path
# only — the envelope enforces S % 128 == 0, so the ragged-edge
# machinery its threadgroup staging exists for never runs):
#   - Q/K/V fragments are read DIRECTLY from device memory (ccv reads
#     K and V directly too; it stages only Q — per c-block! — and O
#     through the threadgroup block, because async_copy is its edge
#     clamp). No functional threadgroup memory -> ZERO barriers in
#     the main loop; a dummy allocation remains as a residency
#     throttle (see the kernel comment — worth 36% at S=8192 D=128).
#   - Q caching follows ccv's D<=96 table policy, but the D=128
#     streamed reload comes straight from device (L1-resident row),
#     not smem — measured: full-cache at D=128 is 20% slower even
#     with residency throttled (register cliff), 2.3x unthrottled.
#   - 1-D q-tile-major grid like ccv (a (q_tile, head) 2-D grid ran
#     2.5x slower at S=8192 D=128 — 16 desynced K/V streams).
#   - P overwrites S in place (ccv keeps separate arrays).
#   - O direct fragment stores (2xf32 = 8 B); L in natural log, f32.
#
# Timing: wall clock around `--dispatches` enqueues + one synchronize,
# divided out (the verified-reliable bracket on this toolchain). This
# INCLUDES mojo's per-enqueue dispatch overhead (~160 us measured on a
# noop), which the references' command-buffer GPU time excludes — a
# deliberate, conservative penalty against mojo; negligible at the
# canonical multi-head shapes (>= 2 ms/iteration).
#
# AIR dump: every run rewrites /tmp/mojo_fwd_metal_d{64,128}.air.ll
# (textual AIR LLVM IR, the Metal PTX analog); master_bench.py (metal)
# copies it into air/ and runs the op-mix diff vs reference_air/.

from std.gpu import barrier, block_idx, lane_id, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.math import ceildiv, exp, exp2, log, sqrt
from std.memory import stack_allocation
from std.sys import argv, llvm_intrinsic
from std.time import perf_counter_ns

comptime BR = 16  # q rows per threadgroup (parallelization dimension)
comptime BC = 128  # kv rows per traversal block
comptime TPB = 64  # 2 simdgroups; each owns 8 q rows
comptime LOG2E = 1.4426950408889634


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


def fwd_kernel[
    D: Int
](
    q_g: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k_g: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_g: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    o_g: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    l_g: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    seq: Int,
):
    comptime ND = D // 8  # Q/O fragments per lane (D-dim tiles)
    comptime NC = BC // 8  # S/P fragments per lane (kv-dim tiles)
    comptime HB = 32  # head-block: D-chunk per Q reload when not cached
    # Q register-caching policy — mirrors ccv/MFA's table: cache at
    # D<=96 only. At D=128 the extra 16 live fragments push the kernel
    # past the register cliff (measured: the D=128 no-load skeleton runs
    # 2.3x off the MMA roofline with Q cached, at roofline without).
    comptime Q_CACHED = D <= 96
    # (comptime sqrt would lower to the AIR intrinsic, which the comptime
    # interpreter can't evaluate — hardcode 1/sqrt(D) per variant.)
    comptime inv_sqrt_d = 0.125 if D == 64 else 0.08838834764831845
    comptime scale_log2 = Float32(LOG2E * inv_sqrt_d)

    # Residency throttle: an (otherwise unused) threadgroup allocation
    # caps resident threadgroups at 3/core (32 KiB / ~10.7 KiB). The
    # kernel uses no threadgroup memory, so unthrottled residency runs
    # much higher than ccv's (8 KiB -> 4/core) — and every extra
    # resident threadgroup is another desynced reader of the K/V
    # stream, blowing the instantaneous working set past the SLC right
    # when per-head K+V reaches 4 MiB (S=8192, D=128: 2618 -> 3555
    # GFLOPS from this alone; sweep: 4 KiB=201 ms, 8 KiB=163 ms,
    # 10.7 KiB=154.5 ms, 16 KiB=160 ms).
    var residency_pad = stack_allocation[
        5460, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()

    var lane = Int(lane_id())
    var sidx = Int(thread_idx.x) // 32
    var fl = morton_order(lane)
    var frow = fl[0]
    var fcol = fl[1]
    # 1-D grid, q-tile-major like ccv: consecutive threadgroup ids are
    # q tiles of the SAME head, so the ~concurrently-resident wave
    # shares one K/V stream (SLC/L2 hits). A (q_tile, head) 2-D grid
    # measured 2.5x slower at S=8192 D=128 — 16 desynced K/V streams
    # go DRAM-bound.
    var n_q_tiles = seq // BR
    var q_tile = Int(block_idx.x) % n_q_tiles
    var head = Int(block_idx.x) // n_q_tiles

    # This lane's q row (each simdgroup owns 8 rows of the 16-row tile).
    var row = q_tile * BR + sidx * 8 + frow

    var q_head = q_g + head * seq * D
    var k_head = k_g + head * seq * D
    var v_head = v_g + head * seq * D

    # Per-lane base pointers: every fragment load below is base +
    # comptime offset (folded into the load's addressing), so the
    # unrolled MMA nests carry no per-load index arithmetic.
    var q_base = q_head + row * D + fcol
    var k_lane = k_head + frow  # + kv*D + comptime d offset
    var v_lane = v_head + fcol  # + kv*D + comptime d offset

    # --- Q register-cached for the whole kernel (raw f16, unscaled:
    # like ccv, the scale is folded into the softmax exp2 instead) ---
    var q_frags = InlineArray[
        SIMD[DType.float16, 2], ND if Q_CACHED else 1
    ](uninitialized=True)
    comptime if Q_CACHED:
        comptime for di in range(ND):
            q_frags[di] = q_base.load[width=2](di * 8)

    # Online-softmax state, per lane (lanes sharing a fragment row hold
    # identical values). m is the scaled log2-domain running max.
    var m = Float32(-3.0e38)
    var l = Float32(0)
    var o_frags = InlineArray[SIMD[DType.float32, 2], ND](
        fill=SIMD[DType.float32, 2](0)
    )

    for cb in range(seq // BC):
        var c = cb * BC

        # --- S = Q·K^T (raw, f16 accumulate; 8x8 MMAs) ---
        # B fragments are K^T: element (d, kv) = K[kv*D + d] — two
        # D-strided scalar loads per fragment, exactly ccv's
        # transposed device load.
        var s_frags = InlineArray[SIMD[DType.float16, 2], NC](
            fill=SIMD[DType.float16, 2](0)
        )
        var k_blk = k_lane + (c + fcol) * D
        comptime if Q_CACHED:
            comptime for di in range(ND):
                comptime for ci in range(NC):
                    var b = SIMD[DType.float16, 2](
                        k_blk.load(ci * 8 * D + di * 8),
                        k_blk.load(ci * 8 * D + di * 8 + D),
                    )
                    s_frags[ci] = mma8x8_f16(q_frags[di], b, s_frags[ci])
        else:
            # Q streamed in HB-column chunks (ccv's D=128 policy), but
            # straight from device — no smem staging, no barriers.
            for d_outer in range(0, D, HB):
                var qc = InlineArray[SIMD[DType.float16, 2], HB // 8](
                    uninitialized=True
                )
                comptime for di in range(HB // 8):
                    qc[di] = q_base.load[width=2](d_outer + di * 8)
                var k_chunk = k_blk + d_outer
                comptime for di in range(HB // 8):
                    comptime for ci in range(NC):
                        var b = SIMD[DType.float16, 2](
                            k_chunk.load(ci * 8 * D + di * 8),
                            k_chunk.load(ci * 8 * D + di * 8 + D),
                        )
                        s_frags[ci] = mma8x8_f16(qc[di], b, s_frags[ci])

        # --- online softmax (log2 domain, scale folded into exp2) ---
        var mx16 = s_frags[0]
        comptime for ci in range(1, NC):
            mx16 = max(mx16, s_frags[ci])
        var m_tile = max(Float32(mx16[0]), Float32(mx16[1]))
        m_tile = max(m_tile, warp.shuffle_xor(m_tile, UInt32(1)))
        m_tile = max(m_tile, warp.shuffle_xor(m_tile, UInt32(8)))
        var m_new = m_tile * scale_log2

        var corr = Float32(1)
        if m_new > m:
            corr = exp2(m - m_new)
            m = m_new

        # P overwrites S in place (f16, same fragment layout == the
        # A-operand layout, so the PV MMAs consume it directly). l sums
        # the ROUNDED f16 P (like ccv) so O's normalization matches PV.
        var l_pair = SIMD[DType.float32, 2](0)
        comptime for ci in range(NC):
            var p = exp2(
                s_frags[ci].cast[DType.float32]() * scale_log2 - m
            ).cast[DType.float16]()
            s_frags[ci] = p
            l_pair += p.cast[DType.float32]()
        var l_new = l_pair[0] + l_pair[1]
        l_new += warp.shuffle_xor(l_new, UInt32(1))
        l_new += warp.shuffle_xor(l_new, UInt32(8))
        l = l * corr + l_new

        # --- O = O*corr + P·V (f32 accumulate) ---
        # V fragments are natural-orientation: element (kv, d) =
        # V[kv*D + d] — one contiguous 2xf16 load per fragment.
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
        residency_pad.store(lane, Float16(l))
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
        # m is the scaled log2-domain max; convert to natural log.
        l_g[head * seq + row] = (m + log2_f32(l)) / Float32(LOG2E)


@always_inline
def log2_f32(x: Float32) -> Float32:
    return log(x) * Float32(LOG2E)


def run_bench[
    D: Int, dump_path: StaticString
](
    ctx: DeviceContext,
    qp: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    kp: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    vp: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    op: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    lp: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    seq: Int,
    heads: Int,
    iters: Int,
    warmup: Int,
    dispatches: Int,
) raises -> List[Float64]:
    var compiled = ctx.compile_function[
        fwd_kernel[D], dump_asm=dump_path
    ]()
    var grid_x = ceildiv(seq, BR) * heads  # 1-D, q-tile-major (see kernel)
    for _ in range(warmup):
        ctx.enqueue_function(
            compiled, qp, kp, vp, op, lp, seq,
            grid_dim=grid_x, block_dim=TPB,
        )
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        for _ in range(dispatches):
            ctx.enqueue_function(
                compiled, qp, kp, vp, op, lp, seq,
                grid_dim=grid_x, block_dim=TPB,
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        times.append(Float64(t1 - t0) / Float64(dispatches) / 1000.0)
    return times^


def main() raises:
    var seq = 4096
    var head_dim = 128
    var heads = 16
    var iters = 10
    var warmup = 3
    var dispatches = 5
    var check = False

    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--" or a == "":  # `mojo run file.mojo -- args` passes the --
            i += 1
            continue
        if a == "--check":
            check = True
            i += 1
            continue
        i += 1
        if i >= len(args):
            raise Error("missing value for " + a)
        var v = Int(String(args[i]))
        if a == "--seq":
            seq = v
        elif a == "--head-dim":
            head_dim = v
        elif a == "--heads":
            heads = v
        elif a == "--iters":
            iters = v
        elif a == "--warmup":
            warmup = v
        elif a == "--dispatches":
            dispatches = v
        else:
            raise Error("unknown flag: " + a)
        i += 1
    if head_dim != 64 and head_dim != 128:
        raise Error("--head-dim must be 64 or 128 (v1 envelope)")
    if seq % BC != 0:
        raise Error("--seq must be a multiple of 128 (v1 envelope)")

    var ctx = DeviceContext()
    var n = heads * seq * head_dim
    var q_buf = ctx.enqueue_create_buffer[DType.float16](n)
    var k_buf = ctx.enqueue_create_buffer[DType.float16](n)
    var v_buf = ctx.enqueue_create_buffer[DType.float16](n)
    var o_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var l_buf = ctx.enqueue_create_buffer[DType.float32](heads * seq)

    # Deterministic inputs in [-0.5, 0.5) — same generator family and
    # per-head seeds as the reference CLIs (xorshift64).
    for which in range(3):
        var buf = q_buf if which == 0 else (k_buf if which == 1 else v_buf)
        with buf.map_to_host() as h:
            for hd in range(heads):
                var state = (
                    UInt64(42 + which + 3 * hd)
                    * UInt64(6364136223846793005)
                    + UInt64(1442695040888963407)
                )
                var base = hd * seq * head_dim
                for e in range(seq * head_dim):
                    state ^= state << 13
                    state ^= state >> 7
                    state ^= state << 17
                    h[base + e] = Float16(
                        Float64(state % 1_000_000) / 1_000_000 - 0.5
                    )

    var qp = q_buf.unsafe_ptr()
    var kp = k_buf.unsafe_ptr()
    var vp = v_buf.unsafe_ptr()
    var op = o_buf.unsafe_ptr()
    var lp = l_buf.unsafe_ptr()

    var times_us: List[Float64]
    if head_dim == 64:
        times_us = run_bench[
            64, StaticString("/tmp/mojo_fwd_metal_d64.air.ll")
        ](ctx, qp, kp, vp, op, lp, seq, heads, iters, warmup, dispatches)
    else:
        times_us = run_bench[
            128, StaticString("/tmp/mojo_fwd_metal_d128.air.ll")
        ](ctx, qp, kp, vp, op, lp, seq, heads, iters, warmup, dispatches)

    # --- aggregate (insertion sort; tiny lists) ---
    var sorted_us = List[Float64]()
    for t in times_us:
        sorted_us.append(t)
    for a_i in range(1, len(sorted_us)):
        var key = sorted_us[a_i]
        var b_i = a_i - 1
        while b_i >= 0 and sorted_us[b_i] > key:
            sorted_us[b_i + 1] = sorted_us[b_i]
            b_i -= 1
        sorted_us[b_i + 1] = key
    var min_us = sorted_us[0]
    var median_us = sorted_us[len(sorted_us) // 2]
    var flops = (
        4.0 * Float64(seq) * Float64(seq) * Float64(head_dim) * Float64(heads)
    )

    # --- optional CPU reference check (first 2 heads, strided rows) ---
    var max_err = Float64(-1.0)
    if check:
        max_err = 0.0
        var scale = 1.0 / sqrt(Float64(head_dim))
        with q_buf.map_to_host() as qh:
            with k_buf.map_to_host() as kh:
                with v_buf.map_to_host() as vh:
                    with o_buf.map_to_host() as oh:
                        for hd in range(min(2, heads)):
                            var base = hd * seq * head_dim
                            var row_stride = max(1, seq // 64)
                            for r in range(0, seq, row_stride):
                                var scores = List[Float64]()
                                var m = Float64(-1e30)
                                for c in range(seq):
                                    var dot = Float64(0)
                                    for d in range(head_dim):
                                        dot += Float64(
                                            qh[base + r * head_dim + d]
                                        ) * Float64(kh[base + c * head_dim + d])
                                    var sc = dot * scale
                                    scores.append(sc)
                                    m = max(m, sc)
                                var l = Float64(0)
                                for c in range(seq):
                                    scores[c] = exp(scores[c] - m)
                                    l += scores[c]
                                for d in range(head_dim):
                                    var acc = Float64(0)
                                    for c in range(seq):
                                        acc += scores[c] * Float64(
                                            vh[base + c * head_dim + d]
                                        )
                                    var expected = acc / l
                                    var actual = Float64(
                                        oh[base + r * head_dim + d]
                                    )
                                    max_err = max(
                                        max_err, abs(expected - actual)
                                    )

    # --- one JSON line on stdout ---
    var out = String('{"impl":"mojo","kernel":"forward","seq":') + String(seq)
    out += String(',"heads":') + String(heads)
    out += String(',"head_dim":') + String(head_dim)
    out += String(',"dtype":"fp16","iters":') + String(iters)
    out += String(',"warmup":') + String(warmup)
    out += String(',"dispatches_per_cb":') + String(dispatches)
    out += String(',"timing":"wall_enqueue_sync"')
    out += String(',"gpu_time_us":[')
    for t_i in range(len(times_us)):
        if t_i > 0:
            out += String(",")
        out += String(times_us[t_i])
    out += String('],"min_us":') + String(min_us)
    out += String(',"median_us":') + String(median_us)
    out += String(',"gflops_4rcd":') + String(flops / (min_us * 1000.0))
    if check:
        out += String(',"check_max_error":') + String(max_err)
    out += String("}")
    print(out)
