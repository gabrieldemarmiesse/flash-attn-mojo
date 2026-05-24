"""Launch helper for the fwd kernel.

Reconstructs a non-owning `DeviceContext` from a cached handle, computes
the dynamic-smem budget (q + k + v + scratch), JIT-compiles + enqueues
`fwd_kernel`.

For seqlens not aligned to `kBlockN`, we pad Q/K/V/O up to a multiple of
`kBlockN` in device memory and run the kernel on those, then copy the
first `seqlen` rows of O back to the user buffer. This sidesteps the
masked-LayoutTensor partial-tile path which triggers IMAs in some
configs. The kernel receives `padded_seq_len` (used by the tile loop)
and `actual_seq_len` (used for per-element score masking against keys
in the padding tail).
"""

from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import size_of

from kernel import fwd_kernel
from common import kNThreads, kBlockM, kBlockN, kBlockK, kWM, kWN


def launch_fwd[
    dtype: DType,
    head_dim: Int,
    causal: Bool,
    use_external_stream: Bool,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    softmax_scale: Float32,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    o_addr: Int,
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
    nheads_kv_int: Int,
    softcap: Float32,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutExternalOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    # Dynamic smem budget (bytes):
    #   Q tile:     BM * head_dim * size_of[dtype]
    #   K tile:     BN * head_dim * size_of[dtype]
    #   V tile:     BN * BN       * size_of[dtype]   (BN == head_dim for our shapes)
    #   scratch:    2 * num_warps_n * BM * size_of[accum=fp32]    (zero when num_warps_n == 1)
    # plus the output write-back reuses q_smem in place — same buffer, no extra.
    comptime q_bytes: Int = kBlockM * head_dim * size_of[dtype]()
    comptime k_bytes: Int = kBlockN * head_dim * size_of[dtype]()
    comptime v_bytes: Int = kBlockN * kBlockN * size_of[dtype]()
    # `p_smem` is reserved unconditionally so `multistage_mma`'s
    # a_smem_iter parameter (which must be in SHARED address space)
    # type-checks even when num_warps_n == 1 and P stays in registers.
    comptime p_bytes: Int = kBlockM * kBlockN * size_of[dtype]()
    comptime scratch_bytes: Int = 0  # num_warps_n == 1 ⇒ warp_scratch is empty
    comptime smem_bytes: Int = (
        q_bytes + k_bytes + v_bytes + p_bytes + scratch_bytes
    )

    var compiled = ctx.compile_function[
        fwd_kernel[dtype, head_dim, causal],
        fwd_kernel[dtype, head_dim, causal],
    ](func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_bytes))

    var padded_seqlen: Int = ceildiv(seqlen_int, Int(kBlockN)) * Int(kBlockN)
    var needs_pad: Bool = padded_seqlen != seqlen_int

    # Whether we run the kernel against the user's tensors directly, or
    # against the padded buffers, the input ptrs/strides handed to the
    # kernel are computed below.
    var k_q_addr: Int = q_addr
    var k_k_addr: Int = k_addr
    var k_v_addr: Int = v_addr
    var k_o_addr: Int = o_addr
    var k_q_b: Int = q_b_stride
    var k_q_l: Int = q_l_stride
    var k_q_h: Int = q_h_stride
    var k_k_b: Int = k_b_stride
    var k_k_l: Int = k_l_stride
    var k_k_h: Int = k_h_stride
    var k_v_b: Int = v_b_stride
    var k_v_l: Int = v_l_stride
    var k_v_h: Int = v_h_stride
    var k_o_b: Int = o_b_stride
    var k_o_l: Int = o_l_stride
    var k_o_h: Int = o_h_stride

    # Padding requires the user tensors to be contiguous along the
    # (L, D) inner dims so each per-(batch, head) plane can be moved
    # as one packed (seqlen × head_dim) blob. Higher-batch/head
    # callers with non-contiguous strides aren't on our hot path
    # (torch's default is contiguous); fail loud if assumed otherwise.
    if needs_pad:
        if q_l_stride != head_dim or k_l_stride != head_dim or v_l_stride != head_dim or o_l_stride != head_dim:
            raise Error(
                "flash_attn_mojo fwd: unaligned seqlen requires contiguous"
                " (L, D) inner strides on Q/K/V/O for the pad-and-copy fast"
                " path. Got non-head_dim l_stride."
            )

    if needs_pad:
        # padded per-(b,h) plane stride
        var pl_stride: Int = head_dim
        var ph_stride: Int = padded_seqlen * head_dim
        var pb_stride: Int = nheads_int * padded_seqlen * head_dim

        var total_elts: Int = batch_int * nheads_int * padded_seqlen * head_dim
        var total_elts_kv: Int = batch_int * nheads_kv_int * padded_seqlen * head_dim

        var q_buf = ctx.enqueue_create_buffer[dtype](total_elts)
        var k_buf = ctx.enqueue_create_buffer[dtype](total_elts_kv)
        var v_buf = ctx.enqueue_create_buffer[dtype](total_elts_kv)
        var o_buf = ctx.enqueue_create_buffer[dtype](total_elts)

        var kv_ph_stride: Int = padded_seqlen * head_dim
        var kv_pb_stride: Int = nheads_kv_int * padded_seqlen * head_dim

        # Zero K/V so padded keys produce zero scores; the kernel's
        # actual_seq_len-based masking sets those P slots to -inf, so the
        # zero V tail just provides safe loads. Q tail / O tail don't
        # affect correctness either since we copy back only the first
        # `seqlen_int` rows of O.
        ctx.enqueue_memset(k_buf, Scalar[dtype](0))
        ctx.enqueue_memset(v_buf, Scalar[dtype](0))

        var q_src = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=q_addr
        )
        var k_src = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=k_addr
        )
        var v_src = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=v_addr
        )

        # Copy per-(batch, head) plane: seqlen_int * head_dim contiguous
        # elements from user ptr to padded buffer ptr. Q uses nheads_q;
        # K/V use nheads_kv (MQA/GQA).
        for b in range(batch_int):
            for h in range(nheads_int):
                var src_off: Int = b * q_b_stride + h * q_h_stride
                var dst_off: Int = b * pb_stride + h * ph_stride
                ctx.enqueue_copy(
                    q_buf.unsafe_ptr() + dst_off,
                    q_src + src_off,
                    seqlen_int * head_dim,
                )
            for h in range(nheads_kv_int):
                var dst_off_kv: Int = b * kv_pb_stride + h * kv_ph_stride
                var src_off_k: Int = b * k_b_stride + h * k_h_stride
                ctx.enqueue_copy(
                    k_buf.unsafe_ptr() + dst_off_kv,
                    k_src + src_off_k,
                    seqlen_int * head_dim,
                )
                var src_off_v: Int = b * v_b_stride + h * v_h_stride
                ctx.enqueue_copy(
                    v_buf.unsafe_ptr() + dst_off_kv,
                    v_src + src_off_v,
                    seqlen_int * head_dim,
                )

        k_q_addr = Int(q_buf.unsafe_ptr())
        k_k_addr = Int(k_buf.unsafe_ptr())
        k_v_addr = Int(v_buf.unsafe_ptr())
        k_o_addr = Int(o_buf.unsafe_ptr())
        k_q_b = pb_stride
        k_q_l = pl_stride
        k_q_h = ph_stride
        k_k_b = kv_pb_stride
        k_k_l = pl_stride
        k_k_h = kv_ph_stride
        k_v_b = kv_pb_stride
        k_v_l = pl_stride
        k_v_h = kv_ph_stride
        k_o_b = pb_stride
        k_o_l = pl_stride
        k_o_h = ph_stride

    var kernel_seqlen: Int = padded_seqlen if needs_pad else seqlen_int

    var q_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_q_addr
    )
    var k_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_k_addr
    )
    var v_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_v_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=k_o_addr
    )

    var grid = (ceildiv(kernel_seqlen, Int(kBlockM)), nheads_int, batch_int)

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            kernel_seqlen,
            seqlen_int,
            nheads_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            k_q_b,
            k_q_l,
            k_q_h,
            k_k_b,
            k_k_l,
            k_k_h,
            k_v_b,
            k_v_l,
            k_v_h,
            k_o_b,
            k_o_l,
            k_o_h,
            nheads_kv_int,
            softcap,
            grid_dim=grid,
            block_dim=(kNThreads,),
            shared_mem_bytes=smem_bytes,
        )
    else:
        ctx.enqueue_function(
            compiled,
            kernel_seqlen,
            seqlen_int,
            nheads_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            k_q_b,
            k_q_l,
            k_q_h,
            k_k_b,
            k_k_l,
            k_k_h,
            k_v_b,
            k_v_l,
            k_v_h,
            k_o_b,
            k_o_l,
            k_o_h,
            nheads_kv_int,
            softcap,
            grid_dim=grid,
            block_dim=(kNThreads,),
            shared_mem_bytes=smem_bytes,
        )

    # Copy O back from the padded buffer into the user's tensor.
    if needs_pad:
        var pb_stride: Int = nheads_int * padded_seqlen * head_dim
        var ph_stride: Int = padded_seqlen * head_dim
        var o_dst = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=o_addr
        )
        var o_padded_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
            unsafe_from_address=k_o_addr
        )
        for b in range(batch_int):
            for h in range(nheads_int):
                var dst_off: Int = b * o_b_stride + h * o_h_stride
                var src_off: Int = b * pb_stride + h * ph_stride
                ctx.enqueue_copy(
                    o_dst + dst_off,
                    o_padded_ptr + src_off,
                    seqlen_int * head_dim,
                )

    comptime if not use_external_stream:
        ctx.synchronize()
