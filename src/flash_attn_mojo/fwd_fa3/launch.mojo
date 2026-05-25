"""Launch helper for the FA3 (sm_90+) fwd kernel.

MVP: bf16, head_dim=64, no causal, no MQA, no softcap, no ALiBi,
no window, no dropout, no LSE. Padding is also out of scope — the
MVP requires seqlen to be a multiple of kFa3BlockN (caller checks
this and routes elsewhere if not). All these features land via
follow-up commits.
"""

from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import size_of

from kernel import fwd_fa3_kernel
from common import kFa3NThreads, kFa3BlockM, kFa3BlockN


def launch_fwd_fa3[
    dtype: DType,
    head_dim: Int,
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

    # Smem budget (bf16, head_dim=64, BM=128, BN=128):
    #   Q tile:   BM * head_dim * 2 = 128 * 64 * 2 = 16 KiB
    #   K tile:   BN * head_dim * 2 = 128 * 64 * 2 = 16 KiB  (×N_STAGES pipeline)
    #   V tile:   BN * head_dim * 2 = 128 * 64 * 2 = 16 KiB  (×N_STAGES)
    # With N_STAGES=2: 16 + 32 + 32 = 80 KiB. Plus 16 B for the
    # mbarrier(s). H100 dynamic-smem cap is 228 KiB so plenty of room.
    comptime N_STAGES: Int = 2
    comptime q_bytes: Int = kFa3BlockM * head_dim * size_of[dtype]()
    comptime kv_stage_bytes: Int = kFa3BlockN * head_dim * size_of[dtype]()
    comptime mbar_bytes: Int = 64  # generous mbarrier scratch (8 barriers × 8 B)
    comptime smem_bytes: Int = (
        q_bytes
        + 2 * N_STAGES * kv_stage_bytes  # K and V pipelines
        + mbar_bytes
    )

    var compiled = ctx.compile_function[
        fwd_fa3_kernel[dtype, head_dim],
        fwd_fa3_kernel[dtype, head_dim],
    ](func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_bytes))

    var q_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=q_addr
    )
    var k_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_addr
    )
    var v_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=v_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=o_addr
    )

    var grid = (ceildiv(seqlen_int, Int(kFa3BlockM)), nheads_int, batch_int)

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            seqlen_int,
            nheads_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            q_b_stride,
            q_l_stride,
            q_h_stride,
            k_b_stride,
            k_l_stride,
            k_h_stride,
            v_b_stride,
            v_l_stride,
            v_h_stride,
            o_b_stride,
            o_l_stride,
            o_h_stride,
            grid_dim=grid,
            block_dim=(kFa3NThreads,),
            shared_mem_bytes=smem_bytes,
        )
    else:
        ctx.enqueue_function(
            compiled,
            seqlen_int,
            nheads_int,
            softmax_scale,
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            q_b_stride,
            q_l_stride,
            q_h_stride,
            k_b_stride,
            k_l_stride,
            k_h_stride,
            v_b_stride,
            v_l_stride,
            v_h_stride,
            o_b_stride,
            o_l_stride,
            o_h_stride,
            grid_dim=grid,
            block_dim=(kFa3NThreads,),
            shared_mem_bytes=smem_bytes,
        )

    comptime if not use_external_stream:
        ctx.synchronize()
