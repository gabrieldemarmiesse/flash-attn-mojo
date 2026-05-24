"""Launch helper for the bwd convert-dQ kernel.

Builds the grid over (q_tile, batch, head), reconstructs a non-owning
`DeviceContext` from a cached handle, JIT-compiles the kernel, and
enqueues it. Mirrors `preprocess_launch.mojo`.
"""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer

from convert_dq import bwd_convert_dq_kernel, kConvertBM, kConvertNThreads


def launch_bwd_convert_dq[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    dqaccum_addr: Int,
    dq_addr: Int,
    dqaccum_b_stride: Int,
    dqaccum_h_stride: Int,
    dqaccum_l_stride: Int,
    dq_b_stride: Int,
    dq_l_stride: Int,
    dq_h_stride: Int,
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
        bwd_convert_dq_kernel[dtype, head_dim],
        bwd_convert_dq_kernel[dtype, head_dim],
    ]()

    var dqaccum_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=dqaccum_addr
    )
    var dq_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=dq_addr
    )

    var grid = (
        ceildiv(seqlen_int, Int(kConvertBM)),
        batch_int,
        nheads_int,
    )

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            seqlen_int,
            dqaccum_ptr,
            dq_ptr,
            dqaccum_b_stride,
            dqaccum_h_stride,
            dqaccum_l_stride,
            dq_b_stride,
            dq_l_stride,
            dq_h_stride,
            grid_dim=grid,
            block_dim=(kConvertNThreads,),
        )
    else:
        ctx.enqueue_function(
            compiled,
            seqlen_int,
            dqaccum_ptr,
            dq_ptr,
            dqaccum_b_stride,
            dqaccum_h_stride,
            dqaccum_l_stride,
            dq_b_stride,
            dq_l_stride,
            dq_h_stride,
            grid_dim=grid,
            block_dim=(kConvertNThreads,),
        )

    comptime if not use_external_stream:
        ctx.synchronize()
