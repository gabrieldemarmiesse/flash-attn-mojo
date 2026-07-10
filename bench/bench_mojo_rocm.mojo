# bench_mojo_rocm: v0 flash-attention FORWARD on AMD CDNA GPUs
# (Q,K,V only — non-causal, dense, fp16 in / fp32 out, MHA batched via
# grid.y like the Metal v0 / ccv's kernel).
#
# Run from the repo root:
#   .venv/bin/mojo run bench/bench_mojo_rocm.mojo -- \
#       --seq 4096 --head-dim 128 --heads 16 --iters 10 [--check]
#
# This is the AMD analog of bench/bench_mojo_metal.mojo. v0 is the
# correctness + harness milestone, NOT a fast kernel: it does NOT use
# the CDNA matrix cores (MFMA / `v_mfma_*`). It is a hand-vectorized
# SIMD kernel: lane j of a 64-wide wavefront owns K-row j of a 64-row
# k/v tile for QK^T (SIMD8 fp16 loads, fp32 accum), online softmax
# keeps lane-replicated (m, l) per q row, and PV broadcasts p_j
# lane-by-lane while the 64 lanes split the D columns.
#
# The one structural difference from the Metal v0 is the wavefront
# width: WARP_SIZE is 64 on CDNA (32 on Apple/NVIDIA), so BK = 64, the
# online-softmax max-reduction is 6 shuffle_xor steps (not 5), and each
# lane owns D/64 output columns (CH) instead of D/32.
#
# Timing: wall clock around `--dispatches` enqueues + one synchronize,
# divided out. This INCLUDES mojo's per-enqueue dispatch overhead, which
# the torch reference's kernel-only (roctracer) time excludes — a
# deliberate, conservative penalty against mojo; negligible at the
# canonical multi-head shapes. master_bench.py (rocm) also re-times this
# kernel under rocprofv3 for an apples-to-apples kernel-only number.
#
# AMDGCN ISA dump: every run rewrites /tmp/mojo_fwd_rocm_d{64,128}.s
# (real gfx942 assembly — the direct PTX analog, courtesy of dump_asm);
# master_bench.py (rocm) copies it into asm/ and runs the op-mix diff.

from std.gpu import barrier, block_idx, lane_id, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from std.math import ceildiv, exp, exp2, log, sqrt
from std.memory import stack_allocation
from std.sys import argv
from std.time import perf_counter_ns

comptime WARP = 64  # CDNA wavefront width
comptime BQ = 16  # q rows per threadblock
comptime BK = 64  # k/v rows per smem tile (== wavefront width: lane j owns row j)
comptime TPB = 256  # threads per block (4 wavefronts)
comptime NW = TPB // WARP  # wavefronts per block
comptime ROWS_PER_WARP = BQ // NW
comptime PAD = 8  # fp16 elements of row padding (LDS banking)
comptime LOG2E = 1.4426950408889634


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
    comptime CH = D // WARP  # O columns owned per lane
    comptime KROW = D + PAD  # padded smem row stride (fp16 elements)
    # exp2 domain: p = 2^(s*qk_scale_log2 - m) == e^((q.k)/sqrt(D) - m').
    # (comptime sqrt lowers to an intrinsic the comptime interpreter
    # can't evaluate — hardcode 1/sqrt(D) per variant.)
    comptime inv_sqrt_d = 0.125 if D == 64 else 0.08838834764831845
    comptime qk_scale_log2 = Float32(LOG2E * inv_sqrt_d)

    var q_smem = stack_allocation[
        BQ * KROW, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var k_smem = stack_allocation[
        BK * KROW, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()
    var v_smem = stack_allocation[
        BK * D, Scalar[DType.float16], address_space = AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp_id = tid // WARP
    var head = Int(block_idx.y)
    var q_tile = Int(block_idx.x)

    var q_head = q_g + head * seq * D
    var k_head = k_g + head * seq * D
    var v_head = v_g + head * seq * D

    # --- load this block's Q tile (BQ x D) once, SIMD8 vectors ---
    comptime Q_VECS = BQ * D // 8
    for i in range(tid, Q_VECS, TPB):
        var row = (i * 8) // D
        var col = (i * 8) % D
        var vec = q_head.load[width=8]((q_tile * BQ + row) * D + col)
        q_smem.store(row * KROW + col, vec)

    # --- per-(warp, q-row) online-softmax state, lane-replicated.
    # m kept in the log2 domain (matches the exp2 calls). o_acc is an
    # InlineArray of per-row SIMD chunks: every index is comptime, so
    # SROA keeps the rows in registers.
    var m_i = SIMD[DType.float32, ROWS_PER_WARP](-1e30)
    var l_i = SIMD[DType.float32, ROWS_PER_WARP](0.0)
    var o_acc = InlineArray[SIMD[DType.float32, CH], ROWS_PER_WARP](
        fill=SIMD[DType.float32, CH](0.0)
    )

    var num_tiles = seq // BK
    for tile in range(num_tiles):
        # --- cooperative K/V tile loads ---
        comptime KV_VECS = BK * D // 8
        for i in range(tid, KV_VECS, TPB):
            var row = (i * 8) // D
            var col = (i * 8) % D
            var src = (tile * BK + row) * D + col
            k_smem.store(row * KROW + col, k_head.load[width=8](src))
            v_smem.store(row * D + col, v_head.load[width=8](src))
        barrier()

        comptime for r in range(ROWS_PER_WARP):
            var qrow = warp_id * ROWS_PER_WARP + r

            # s_j = (q[qrow] . k[lane]) * scale * log2(e), fp32 accum
            var acc = SIMD[DType.float32, 8](0.0)
            comptime for d8 in range(D // 8):
                var qv = q_smem.load[width=8](qrow * KROW + d8 * 8)
                var kv = k_smem.load[width=8](lane * KROW + d8 * 8)
                acc += qv.cast[DType.float32]() * kv.cast[DType.float32]()
            var s = acc.reduce_add() * qk_scale_log2

            # online softmax update (every lane holds every value)
            var tile_max = s
            comptime for i in range(6):  # 64-lane reduction
                tile_max = max(
                    tile_max, warp.shuffle_xor(tile_max, UInt32(1 << i))
                )
            var m_new = max(m_i[r], tile_max)
            var corr = exp2(m_i[r] - m_new)
            var p = exp2(s - m_new)
            l_i[r] = l_i[r] * corr + warp.sum(p)
            m_i[r] = m_new
            var o_r = o_acc[r] * corr

            # o[qrow, lane-chunk] += p_j * v[j, lane-chunk] for all 64 j
            comptime for j in range(BK):
                var pj = warp.shuffle_idx(p, UInt32(j))
                var vv = v_smem.load[width=CH](j * D + lane * CH)
                o_r += pj * vv.cast[DType.float32]()
            o_acc[r] = o_r
        barrier()

    # --- epilogue: O = o_acc / l (fp32), L = logsumexp (natural log) ---
    comptime for r in range(ROWS_PER_WARP):
        var qrow = warp_id * ROWS_PER_WARP + r
        var grow = q_tile * BQ + qrow
        var inv_l = 1.0 / l_i[r]
        o_g.store(
            head * seq * D + grow * D + lane * CH,
            o_acc[r] * inv_l,
        )
        if lane == 0:
            # m is a log2-domain max; convert back to natural log.
            l_g[head * seq + grow] = m_i[r] / Float32(LOG2E) + log(l_i[r])


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
    var grid_x = ceildiv(seq, BQ)
    for _ in range(warmup):
        ctx.enqueue_function(
            compiled, qp, kp, vp, op, lp, seq,
            grid_dim=(grid_x, heads), block_dim=TPB,
        )
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(iters):
        var t0 = perf_counter_ns()
        for _ in range(dispatches):
            ctx.enqueue_function(
                compiled, qp, kp, vp, op, lp, seq,
                grid_dim=(grid_x, heads), block_dim=TPB,
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
        raise Error("--head-dim must be 64 or 128 (v0 envelope)")
    if seq % BQ != 0 or seq % BK != 0:
        raise Error("--seq must be a multiple of 64 (v0 envelope)")

    var ctx = DeviceContext()
    var n = heads * seq * head_dim
    var q_buf = ctx.enqueue_create_buffer[DType.float16](n)
    var k_buf = ctx.enqueue_create_buffer[DType.float16](n)
    var v_buf = ctx.enqueue_create_buffer[DType.float16](n)
    var o_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var l_buf = ctx.enqueue_create_buffer[DType.float32](heads * seq)

    # Deterministic inputs in [-0.5, 0.5) — same xorshift64 generator
    # family and per-head seeds as bench_mojo_metal.mojo.
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
            64, StaticString("/tmp/mojo_fwd_rocm_d64.s")
        ](ctx, qp, kp, vp, op, lp, seq, heads, iters, warmup, dispatches)
    else:
        times_us = run_bench[
            128, StaticString("/tmp/mojo_fwd_rocm_d128.s")
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
