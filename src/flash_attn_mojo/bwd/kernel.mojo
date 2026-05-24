"""Flash-attention backward kernel — MVP scalar/SIMD implementation.

Port of Tri Dao's `compute_dq_dk_dv_1colblock`
(`flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h`, lines 81-826).
MVP envelope (locked at variant compile time):

  - dtype = bf16  (fp16 routed via API-boundary cast — same pattern as fwd)
  - head_dim = 64
  - causal: optional (comptime)
  - softcap: optional (runtime)
  - no alibi, window, dropout
  - single seqlen (Q and K share length)

MQA/GQA: supported via the Tri Dao block layout. Grid Y dim is
`nheads_kv`; inside each block we loop over the `group_size =
nheads_q // nheads_kv` Q-heads sharing this KV-head. K/V smem tiles
are loaded once and reused across the inner group loop. dK_acc /
dV_acc registers accumulate contributions from every Q-head in the
group. dQ contributions go to per-Q-head dQaccum slots — no atomic
conflicts between Q-heads.

The bwd is far harder to make bit-correct than the fwd, so this MVP
trades performance for correctness: smem holds *every* per-block tile
(K, V, Q, dO, S/dS, dP) plus per-row LSE / delta scratch, and the
matmuls run as plain thread-parallel loops over fp32 accumulators rather
than via `multistage_mma` tensor-core MMAs. dK_acc / dV_acc live in
registers (one (n, d) element per thread for the (BN×D)=(64×64)=4096
slot count / 128 threads = 32 elements per thread).

Algorithm (one block per (n_block, batch, head); see upstream lines 81-826):

  1. Load K_tile (BN × D) and V_tile (BN × D) into smem once.
  2. Initialize dK_acc, dV_acc registers to 0.
  3. For each q_block in [0, num_q_blocks):
       a. Load Q (BM × D), dO (BM × D) into smem.
       b. Load LSE[q], delta[q] into smem.
       c. S = Q · K^T                                      (BM × BN) fp32
       d. P = exp2((S * softmax_scale - LSE) * log2e)      (BM × BN) fp32
          Stored back in S_smem.
       e. dV_acc += P^T · dO                               (BN × D)
       f. dP = dO · V^T                                    (BM × BN) fp32
       g. dS = P * (dP - delta) * softmax_scale            (BM × BN) fp32
          Stored back overwriting P in S_smem.
       h. dQ_contrib = dS · K   (per (b,h,q,:) atomic-add into dQaccum fp32)
       i. dK_acc += dS^T · Q.
  4. Write dK_acc (cast bf16) and dV_acc (cast bf16) to gmem.

Smem layout (dynamic), sizes shown at head_dim=64 / hd=128:
    K_smem      : BN × D   bf16   = 8 / 16 KiB
    V_smem      : BN × D   bf16   = 8 / 16 KiB
    Q_smem      : BM × D   bf16   = 8 / 16 KiB
    dO_smem     : BM × D   bf16   = 8 / 16 KiB
    S_smem      : BM × BN  fp32   = 16 KiB   (reused for dS after dV update)
    dP_smem     : BM × BN  fp32   = 16 KiB
    LSE_smem    : BM       fp32   = 256 B
    delta_smem  : BM       fp32   = 256 B
    Total                         ~ 64.5 / 96.5 KiB

This sits under the 99 KiB Ada dynamic-smem cap at hd=128. The
softcap local-derivative factor (1 - (s_post/softcap)^2) is
recomputed on the fly in the dS step from `log(p) + lse[m]`
(minus alibi bias) rather than stashed in smem, saving 16 KiB.
"""

from std.math import exp, log, log2, tanh
from std.math.constants import log2e
from std.sys import size_of
from std.utils.index import StaticTuple
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    thread_idx,
)
from std.gpu.memory import AddressSpace, external_memory
from std.memory import stack_allocation
from std.atomic import Atomic

from common import kBwdNThreads, kBwdBlockM, kBwdBlockN


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
    comptime BM: Int = kBwdBlockM
    comptime BN: Int = kBwdBlockN
    comptime D: Int = head_dim
    comptime nthreads: Int = kBwdNThreads

    var tid: Int = Int(thread_idx.x)
    var n_block: Int = Int(block_idx.x)
    var kv_head_idx: Int = Int(block_idx.y)
    var batch: Int = Int(block_idx.z)
    # Number of Q-heads sharing this KV-head. Caller guarantees
    # nheads_q % nheads_kv == 0.
    var group_size: Int = nheads_q // nheads_kv

    var kv_row_base: Int = n_block * BN
    # Out-of-range KV block guard. Launcher rounds up the grid so seqlen
    # padding tail is possible.
    if kv_row_base >= seq_len:
        return

    # ---- Dynamic smem carve-up.
    # Order: K, V (bf16, BN*D each), Q, dO (bf16, BM*D each),
    #        S (fp32, BM*BN), dP (fp32, BM*BN), LSE (fp32, BM), delta (fp32, BM).
    var k_smem = external_memory[
        Scalar[dtype], address_space=AddressSpace.SHARED, alignment=16
    ]()
    var v_smem = k_smem + (BN * D)
    var q_smem = v_smem + (BN * D)
    var do_smem = q_smem + (BM * D)
    var s_smem = (do_smem + (BM * D)).bitcast[Float32]()
    var dp_smem = s_smem + (BM * BN)
    # Softcap local-derivative factor is recomputed on the fly in the dS
    # step from `log(p) + lse[m]` rather than stashed in a dedicated
    # smem buffer — saves 16 KiB and brings hd=128 under the 99 KiB Ada
    # dynamic-smem cap.
    var lse_smem = dp_smem + (BM * BN)
    var delta_smem = lse_smem + BM

    # ---- Per-thread dK_acc, dV_acc register tiles.
    # BN*D / nthreads = 4096 / 128 = 32 fp32 elements per thread for each.
    # Linear flat layout: thread t owns flat indices [t*ELTS .. (t+1)*ELTS).
    comptime ELTS_PER_THREAD: Int = (BN * D) // nthreads  # 32

    var dk_acc = stack_allocation[ELTS_PER_THREAD, Float32]()
    var dv_acc = stack_allocation[ELTS_PER_THREAD, Float32]()
    comptime for i in range(ELTS_PER_THREAD):
        dk_acc[i] = Float32(0)
        dv_acc[i] = Float32(0)

    # ---- Load K, V (cooperative, contiguous-in-D).
    # K_gmem stride: batch, head, l, d. K/V are indexed by the
    # KV-head, NOT the Q-head — multiple Q-heads share one KV-head
    # under MQA/GQA.
    var k_base_off: Int = batch * k_b_stride + kv_head_idx * k_h_stride
    var v_base_off: Int = batch * v_b_stride + kv_head_idx * v_h_stride
    # Each thread loads (BN*D)/nthreads bf16 elements.
    comptime KV_PER_THREAD: Int = (BN * D) // nthreads  # 32
    for i in range(KV_PER_THREAD):
        var flat: Int = i * nthreads + tid
        var row: Int = flat // D
        var col: Int = flat - row * D
        var g_row: Int = kv_row_base + row
        var dst: Int = row * D + col
        if g_row < seq_len:
            k_smem[dst] = (k_ptr + k_base_off + g_row * k_l_stride + col)[0]
            v_smem[dst] = (v_ptr + v_base_off + g_row * v_l_stride + col)[0]
        else:
            k_smem[dst] = Scalar[dtype](0)
            v_smem[dst] = Scalar[dtype](0)

    barrier()

    # ---- Outer loop over q_blocks.
    var num_q_blocks: Int = (seq_len + BM - 1) // BM
    # Causal block-skip lower bound: Q blocks with q_row < kv_row_base are
    # entirely above the diagonal vs this KV block → P = 0 everywhere →
    # contribute nothing to dK/dV/dQ. Skip them.
    var qb_start: Int = 0
    var qb_end: Int = num_q_blocks
    @parameter
    if causal:
        qb_start = (n_block * BN) // BM
    # Sliding-window block-skip. For (q_row, kv_row) to be in window:
    #   kv_row - right <= q_row <= kv_row + left   (when each side >= 0).
    # kv_row spans [kv_row_base, kv_row_base + BN - 1] in this block,
    # so the q-row range that can contribute is
    #   [kv_row_base - right, (kv_row_base + BN - 1) + left]
    # for the right and left sides respectively (when each >= 0).
    var has_window: Bool = window_left >= 0 or window_right >= 0
    if has_window:
        if window_right >= 0:
            # Lower bound on q_row: q_row >= kv_row_base - window_right.
            # Block qb contains q_rows [qb*BM, qb*BM+BM-1]; need
            # qb*BM + BM - 1 >= kv_row_base - window_right, i.e.
            # qb >= ceil((kv_row_base - window_right - BM + 1) / BM).
            var lo: Int = kv_row_base - window_right - BM + 1
            var qb_lo: Int = (lo + BM - 1) // BM if lo > 0 else 0
            # Floor-div for negative numerators: lo <= 0 ⇒ qb_lo = 0.
            if qb_lo > qb_start:
                qb_start = qb_lo
        if window_left >= 0:
            # Upper bound on q_row: q_row <= kv_row_base + BN - 1 + window_left.
            # Block qb contains q_rows [qb*BM, qb*BM+BM-1]; need
            # qb*BM <= kv_row_base + BN - 1 + window_left, i.e.
            # qb <= (kv_row_base + BN - 1 + window_left) // BM.
            var hi_inclusive: Int = (
                kv_row_base + BN - 1 + window_left
            ) // BM
            var qb_hi: Int = hi_inclusive + 1
            if qb_hi < qb_end:
                qb_end = qb_hi

    var scale_f: Float32 = softmax_scale
    var softcap_f: Float32 = softcap
    var has_softcap: Bool = softcap_f > Float32(0)
    var softcap_inv: Float32 = (
        Float32(1) / softcap_f if has_softcap else Float32(0)
    )

    # ---- Outer Q-head-in-group loop (MQA/GQA). For non-MQA configs
    # group_size == 1 so this collapses to a single pass — same flops
    # as before. K/V smem tiles are loaded once before this loop and
    # reused across every Q-head in the group; dK/dV registers
    # accumulate across the full group.
    # ---- ALiBi slope load. `alibi_ptr` is null (addr 0) when caller
    # passes no slopes — the fast path stays zero-overhead. Slope is
    # per (batch, q-head); we load it inside the q-head loop below.
    var has_alibi: Bool = Int(alibi_ptr) != 0

    for h_q_in_group in range(group_size):
        var q_head_idx: Int = kv_head_idx * group_size + h_q_in_group
        var q_base_off: Int = batch * q_b_stride + q_head_idx * q_h_stride
        var alibi_slope: Float32 = Float32(0)
        if has_alibi:
            var alibi_off: Int = (
                batch * alibi_b_stride + q_head_idx * alibi_h_stride
            )
            alibi_slope = (alibi_ptr + alibi_off)[0]
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

        for qb in range(qb_start, qb_end):
            var q_row_base: Int = qb * BM

            # ---- Load Q, dO into smem.
            for i in range(KV_PER_THREAD):
                var flat: Int = i * nthreads + tid
                var row: Int = flat // D
                var col: Int = flat - row * D
                var g_row: Int = q_row_base + row
                var dst: Int = row * D + col
                if g_row < seq_len:
                    q_smem[dst] = (
                        q_ptr + q_base_off + g_row * q_l_stride + col
                    )[0]
                    do_smem[dst] = (
                        do_ptr + do_base_off + g_row * do_l_stride + col
                    )[0]
                else:
                    q_smem[dst] = Scalar[dtype](0)
                    do_smem[dst] = Scalar[dtype](0)

            # ---- Load LSE, delta (BM elements each, 128 threads → 2 per thread
            #      at BM=64 first thread of each pair does it).
            if tid < BM:
                var g_row: Int = q_row_base + tid
                if g_row < seq_len:
                    lse_smem[tid] = (lse_ptr + lse_base_off + g_row)[0]
                    delta_smem[tid] = (delta_ptr + delta_base_off + g_row)[0]
                else:
                    # Padded rows: LSE = -inf, delta = 0 → P=0, contributes nothing.
                    lse_smem[tid] = Float32(-1.0e30)
                    delta_smem[tid] = Float32(0)

            barrier()

            # ---- S = Q · K^T  (BM × BN), fp32.
            # Output layout S_smem[m, n] at index m*BN + n. 128 threads, BM*BN=4096
            # cells → 32 cells per thread, flat-strided.
            comptime SBN_PER_THREAD: Int = (BM * BN) // nthreads  # 32
            for i in range(SBN_PER_THREAD):
                var flat: Int = i * nthreads + tid
                var m: Int = flat // BN
                var n: Int = flat - m * BN
                var acc: Float32 = Float32(0)
                for d in range(D):
                    var qv: Float32 = q_smem[m * D + d].cast[DType.float32]()
                    var kv: Float32 = k_smem[n * D + d].cast[DType.float32]()
                    acc = acc + qv * kv
                s_smem[m * BN + n] = acc

            barrier()

            # ---- Apply scale, subtract LSE, exponentiate to P. Store back in S_smem.
            # P[m, n] = exp(S[m, n] * softmax_scale - LSE[m]).
            # Use natural exp here (delta convention matches: dS = P * (dP - delta)
            # with natural-log LSE — same as the pytorch fallback).
            for i in range(SBN_PER_THREAD):
                var flat: Int = i * nthreads + tid
                var m: Int = flat // BN
                var n: Int = flat - m * BN
                var g_row: Int = q_row_base + m
                var g_col: Int = kv_row_base + n
                # s_pre = (Q · K^T) * softmax_scale.
                var s_pre: Float32 = s_smem[m * BN + n] * scale_f
                # Apply softcap (if enabled). s_post is the value that
                # actually feeds softmax: s_post = softcap*tanh(s_pre/softcap).
                # Local derivative ds_post/ds_pre = 1 - (s_post/softcap)^2
                # is recomputed in the dS loop from `log(p) + lse[m]`
                # rather than stashed here — saves 16 KiB of smem (lets
                # hd=128 fit under the 99 KiB Ada dynamic-smem cap).
                var s_post: Float32 = s_pre
                if has_softcap:
                    s_post = softcap_f * tanh(s_pre * softcap_inv)
                # ALiBi bias is additive and matches the fwd kernel
                # convention exactly: causal=+slope*col, non-causal=
                # -slope*|row-col|. Composition order: scores_raw →
                # softcap → alibi → mask → exp(... - LSE). Pure additive
                # so dS/dS_pre_alibi = 1; doesn't affect softcap
                # derivative below.
                if has_alibi:
                    @parameter
                    if causal:
                        s_post = s_post + alibi_slope * Float32(g_col)
                    else:
                        var d_rc: Int = g_col - g_row
                        if d_rc < 0:
                            d_rc = -d_rc
                        s_post = s_post - alibi_slope * Float32(d_rc)
                var p: Float32
                var masked: Bool = g_row >= seq_len or g_col >= seq_len
                @parameter
                if causal:
                    if g_row < g_col:
                        masked = True
                # Sliding-window mask: kv_col must lie in
                # [g_row - window_left, g_row + window_right] when each
                # bound is >= 0. -1 = unbounded.
                if has_window:
                    if window_left >= 0 and g_col < g_row - window_left:
                        masked = True
                    if window_right >= 0 and g_col > g_row + window_right:
                        masked = True
                if masked:
                    p = Float32(0)
                else:
                    p = exp(s_post - lse_smem[m])
                s_smem[m * BN + n] = p

            barrier()

            # ---- dV_acc += P^T · dO   (BN × D), accumulate in registers.
            # dV[n, d] += sum_m P[m, n] * dO[m, d].
            # Thread t owns dv_acc indices [t*32 .. (t+1)*32). Flat → (n, d).
            for i in range(ELTS_PER_THREAD):
                var flat: Int = tid * ELTS_PER_THREAD + i
                var n: Int = flat // D
                var d: Int = flat - n * D
                var acc: Float32 = Float32(0)
                for m in range(BM):
                    var pv: Float32 = s_smem[m * BN + n]
                    var dov: Float32 = do_smem[m * D + d].cast[DType.float32]()
                    acc = acc + pv * dov
                dv_acc[i] = dv_acc[i] + acc

            # ---- dP = dO · V^T  (BM × BN), fp32.
            for i in range(SBN_PER_THREAD):
                var flat: Int = i * nthreads + tid
                var m: Int = flat // BN
                var n: Int = flat - m * BN
                var acc: Float32 = Float32(0)
                for d in range(D):
                    var dov: Float32 = do_smem[m * D + d].cast[DType.float32]()
                    var vv: Float32 = v_smem[n * D + d].cast[DType.float32]()
                    acc = acc + dov * vv
                dp_smem[m * BN + n] = acc

            barrier()

            # ---- dS = P * (dP - delta[m]) * softmax_scale. Overwrites P in S_smem.
            # We fold softmax_scale into dS here (upstream FA2 splits the scale
            # between the dQ and dK paths; the simplest correct accounting is to
            # bake the full scale into dS once so the trailing matmuls' results
            # are already scaled — matches the pytorch fallback in _fn.py which
            # multiplies dq/dk by softmax_scale after a `ds_raw = ds_post * ...`
            # step).
            for i in range(SBN_PER_THREAD):
                var flat: Int = i * nthreads + tid
                var m: Int = flat // BN
                var n: Int = flat - m * BN
                var p: Float32 = s_smem[m * BN + n]
                var dpv: Float32 = dp_smem[m * BN + n]
                # dS_post = P * (dP - delta). Chain through softcap:
                # dS_pre = dS_post * (1 - (s_post/softcap)^2). Then bake
                # softmax_scale into dS so the trailing matmuls land
                # dq/dk pre-scaled (matches the pytorch fallback order).
                var ds: Float32 = p * (dpv - delta_smem[m])
                if has_softcap:
                    # Reconstruct the post-softcap (pre-alibi) s_post from
                    # p and LSE: p = exp(s_post_with_alibi - lse[m]), so
                    # s_post = log(p) + lse[m] - alibi_bias. ALiBi is
                    # additive so we subtract it back out to get the
                    # pre-alibi s_post that softcap saw. When p == 0
                    # (masked) ds is already 0 — skip log(0).
                    if p > Float32(0):
                        var g_row: Int = q_row_base + m
                        var g_col: Int = kv_row_base + n
                        var s_post: Float32 = log(p) + lse_smem[m]
                        if has_alibi:
                            @parameter
                            if causal:
                                s_post = s_post - alibi_slope * Float32(g_col)
                            else:
                                var d_rc: Int = g_col - g_row
                                if d_rc < 0:
                                    d_rc = -d_rc
                                s_post = s_post + alibi_slope * Float32(d_rc)
                        var t: Float32 = s_post * softcap_inv
                        ds = ds * (Float32(1) - t * t)
                ds = ds * scale_f
                s_smem[m * BN + n] = ds

            barrier()

            # ---- dQ_contrib (BM × D) = dS · K. Atomic-add into dqaccum.
            # Thread layout: BM*D = 4096 cells / 128 threads = 32 per thread.
            for i in range(ELTS_PER_THREAD):
                var flat: Int = i * nthreads + tid
                var m: Int = flat // D
                var d: Int = flat - m * D
                var g_row: Int = q_row_base + m
                if g_row < seq_len:
                    var acc: Float32 = Float32(0)
                    for n in range(BN):
                        var dsv: Float32 = s_smem[m * BN + n]
                        var kv: Float32 = k_smem[n * D + d].cast[DType.float32]()
                        acc = acc + dsv * kv
                    var dqa_addr = (
                        dqaccum_ptr
                        + dqa_base_off
                        + g_row * dqa_l_stride
                        + d
                    )
                    _ = Atomic.fetch_add(dqa_addr, acc)

            # ---- dK_acc += dS^T · Q  (BN × D). Accumulate in registers.
            for i in range(ELTS_PER_THREAD):
                var flat: Int = tid * ELTS_PER_THREAD + i
                var n: Int = flat // D
                var d: Int = flat - n * D
                var acc: Float32 = Float32(0)
                for m in range(BM):
                    var dsv: Float32 = s_smem[m * BN + n]
                    var qv: Float32 = q_smem[m * D + d].cast[DType.float32]()
                    acc = acc + dsv * qv
                dk_acc[i] = dk_acc[i] + acc

            barrier()

    # ---- Write dK, dV back to gmem (cast to bf16). dK has softmax_scale
    # already folded in (since we baked it into dS). dV does NOT have it.
    var dk_base_off: Int = batch * dk_b_stride + kv_head_idx * dk_h_stride
    var dv_base_off: Int = batch * dv_b_stride + kv_head_idx * dv_h_stride
    for i in range(ELTS_PER_THREAD):
        var flat: Int = tid * ELTS_PER_THREAD + i
        var n: Int = flat // D
        var d: Int = flat - n * D
        var g_row: Int = kv_row_base + n
        if g_row < seq_len:
            (dk_ptr + dk_base_off + g_row * dk_l_stride + d)[0] = (
                dk_acc[i].cast[dtype]()
            )
            (dv_ptr + dv_base_off + g_row * dv_l_stride + d)[0] = (
                dv_acc[i].cast[dtype]()
            )
