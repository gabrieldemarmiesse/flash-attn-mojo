"""Launch helper for the bwd-preprocess kernel.

Builds the grid over (q_tile, head, batch), reconstructs a non-owning
`DeviceContext` from a cached handle, JIT-compiles the kernel, and
enqueues it.
"""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer

from preprocess import bwd_preprocess_kernel, kPreprocBM, kPreprocNThreads


def launch_bwd_preprocess[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    dout_addr: Int,
    o_addr: Int,
    delta_addr: Int,
    dqaccum_addr: Int,
    do_b_stride: Int,
    do_l_stride: Int,
    do_h_stride: Int,
    o_b_stride: Int,
    o_l_stride: Int,
    o_h_stride: Int,
    delta_b_stride: Int,
    delta_h_stride: Int,
    dqaccum_b_stride: Int,
    dqaccum_h_stride: Int,
    dqaccum_l_stride: Int,
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

    var compiled = ctx.compile_function[
        bwd_preprocess_kernel[dtype, head_dim],
        bwd_preprocess_kernel[dtype, head_dim],
    ]()

    var dout_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=dout_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=o_addr
    )
    var delta_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=delta_addr
    )
    var dqaccum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dqaccum_addr
    )

    var grid = (
        ceildiv(seqlen_int, Int(kPreprocBM)),
        nheads_int,
        batch_int,
    )

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            seqlen_int,
            nheads_int,
            dout_ptr,
            o_ptr,
            delta_ptr,
            dqaccum_ptr,
            do_b_stride,
            do_l_stride,
            do_h_stride,
            o_b_stride,
            o_l_stride,
            o_h_stride,
            delta_b_stride,
            delta_h_stride,
            dqaccum_b_stride,
            dqaccum_h_stride,
            dqaccum_l_stride,
            grid_dim=grid,
            block_dim=(kPreprocNThreads,),
        )
    else:
        ctx.enqueue_function(
            compiled,
            seqlen_int,
            nheads_int,
            dout_ptr,
            o_ptr,
            delta_ptr,
            dqaccum_ptr,
            do_b_stride,
            do_l_stride,
            do_h_stride,
            o_b_stride,
            o_l_stride,
            o_h_stride,
            delta_b_stride,
            delta_h_stride,
            dqaccum_b_stride,
            dqaccum_h_stride,
            dqaccum_l_stride,
            grid_dim=grid,
            block_dim=(kPreprocNThreads,),
        )

    comptime if not use_external_stream:
        ctx.synchronize()
