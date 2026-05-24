"""Launch helper for the main bwd kernel.

Computes the dynamic-smem budget and enqueues `bwd_kernel`. Grid is
(num_n_blocks, nheads_kv, batch) — one block per (KV-block, KV-head,
batch). The kernel loops over the `group_size = nheads_q // nheads_kv`
Q-heads sharing each KV-head internally. The MVP launches the kernel
directly on the user's tensors (no seqlen padding) since the kernel
internally boundary-masks both K-row and Q-row OOB accesses to zeros.
"""

from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import size_of

from kernel import bwd_kernel
from common import kBwdNThreads, kBwdBlockM, kBwdBlockN


def launch_bwd[
    dtype: DType,
    head_dim: Int,
    causal: Bool,
    use_external_stream: Bool,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_q_int: Int,
    nheads_kv_int: Int,
    softmax_scale: Float32,
    softcap: Float32,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    do_addr: Int,
    lse_addr: Int,
    delta_addr: Int,
    dk_addr: Int,
    dv_addr: Int,
    dqa_addr: Int,
    alibi_addr: Int,
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

    # Smem budget — see kernel.mojo header for the breakdown.
    comptime D: Int = head_dim
    comptime k_bytes: Int = kBwdBlockN * D * size_of[dtype]()
    comptime v_bytes: Int = kBwdBlockN * D * size_of[dtype]()
    comptime q_bytes: Int = kBwdBlockM * D * size_of[dtype]()
    comptime do_bytes: Int = kBwdBlockM * D * size_of[dtype]()
    comptime s_bytes: Int = kBwdBlockM * kBwdBlockN * 4  # fp32
    comptime dp_bytes: Int = kBwdBlockM * kBwdBlockN * 4
    comptime sfcfac_bytes: Int = kBwdBlockM * kBwdBlockN * 4  # fp32 softcap fac
    comptime lse_bytes: Int = kBwdBlockM * 4
    comptime delta_bytes: Int = kBwdBlockM * 4
    comptime smem_bytes: Int = (
        k_bytes + v_bytes + q_bytes + do_bytes
        + s_bytes + dp_bytes + sfcfac_bytes + lse_bytes + delta_bytes
    )

    var compiled = ctx.compile_function[
        bwd_kernel[dtype, head_dim, causal],
        bwd_kernel[dtype, head_dim, causal],
    ](func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)))

    var q_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=q_addr
    )
    var k_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=k_addr
    )
    var v_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=v_addr
    )
    var do_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=do_addr
    )
    var lse_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=lse_addr
    )
    var delta_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=delta_addr
    )
    var dk_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=dk_addr
    )
    var dv_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=dv_addr
    )
    var dqa_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dqa_addr
    )
    var alibi_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=alibi_addr
    )

    var num_n_blocks: Int = ceildiv(seqlen_int, Int(kBwdBlockN))
    var grid = (num_n_blocks, nheads_kv_int, batch_int)

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            seqlen_int,
            nheads_q_int,
            nheads_kv_int,
            softmax_scale,
            softcap,
            q_ptr,
            k_ptr,
            v_ptr,
            do_ptr,
            lse_ptr,
            delta_ptr,
            dk_ptr,
            dv_ptr,
            dqa_ptr,
            alibi_ptr,
            alibi_b_stride,
            alibi_h_stride,
            window_left,
            window_right,
            q_b_stride,
            q_l_stride,
            q_h_stride,
            k_b_stride,
            k_l_stride,
            k_h_stride,
            v_b_stride,
            v_l_stride,
            v_h_stride,
            do_b_stride,
            do_l_stride,
            do_h_stride,
            dk_b_stride,
            dk_l_stride,
            dk_h_stride,
            dv_b_stride,
            dv_l_stride,
            dv_h_stride,
            lse_b_stride,
            lse_h_stride,
            delta_b_stride,
            delta_h_stride,
            dqa_b_stride,
            dqa_h_stride,
            dqa_l_stride,
            grid_dim=grid,
            block_dim=(kBwdNThreads,),
            shared_mem_bytes=smem_bytes,
        )
    else:
        ctx.enqueue_function(
            compiled,
            seqlen_int,
            nheads_q_int,
            nheads_kv_int,
            softmax_scale,
            softcap,
            q_ptr,
            k_ptr,
            v_ptr,
            do_ptr,
            lse_ptr,
            delta_ptr,
            dk_ptr,
            dv_ptr,
            dqa_ptr,
            alibi_ptr,
            alibi_b_stride,
            alibi_h_stride,
            window_left,
            window_right,
            q_b_stride,
            q_l_stride,
            q_h_stride,
            k_b_stride,
            k_l_stride,
            k_h_stride,
            v_b_stride,
            v_l_stride,
            v_h_stride,
            do_b_stride,
            do_l_stride,
            do_h_stride,
            dk_b_stride,
            dk_l_stride,
            dk_h_stride,
            dv_b_stride,
            dv_l_stride,
            dv_h_stride,
            lse_b_stride,
            lse_h_stride,
            delta_b_stride,
            delta_h_stride,
            dqa_b_stride,
            dqa_h_stride,
            dqa_l_stride,
            grid_dim=grid,
            block_dim=(kBwdNThreads,),
            shared_mem_bytes=smem_bytes,
        )

    comptime if not use_external_stream:
        ctx.synchronize()
