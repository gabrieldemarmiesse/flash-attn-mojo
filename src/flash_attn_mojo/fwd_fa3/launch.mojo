"""Launch helper for the FA3 (sm_90+) fwd kernel.

MVP: bf16, head_dim=64, no causal, no MQA, no softcap, no ALiBi,
no window, no dropout, no LSE. Requires contiguous (B, L, H, D)
inputs. Seqlen must be a multiple of kFa3BlockN.

Builds 3D TMA descriptors that view the (B, L, H, D) tensors as
(B*L, H, D) row-major. Each block selects its (b, q_block, h) tile
by passing TMA coords (b * L + q_block * BM, h, 0). H and D are
runtime in the gmem layout; only the smem tile shape is comptime.
"""

from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import size_of
from std.utils.index import IndexList

from layout import UNKNOWN_VALUE
from layout.tma_async import create_split_tma

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

    # Swizzle: SWIZZLE_64B for head_dim*sizeof(bf16) = 128B = 64B*2.
    # Actually 64 bf16 = 128 bytes, so SWIZZLE_128B matches one row.
    # Pick SWIZZLE_64B because it matches the WGMMA tile_layout_k_major
    # K-major atom for D=64 head_dim.
    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # Smem budget: Q + K + V tiles plus mbarriers.
    #   Q: BM × D  bf16 = 128 × 64 × 2 = 16 KiB
    #   K: BN × D  bf16 = 128 × 64 × 2 = 16 KiB
    #   V: BN × D  bf16 = 128 × 64 × 2 = 16 KiB
    # Total: 48 KiB + small mbarrier scratch (well under H100's 228 KiB).
    comptime q_bytes: Int = kFa3BlockM * head_dim * size_of[dtype]()
    comptime k_bytes: Int = kFa3BlockN * head_dim * size_of[dtype]()
    comptime v_bytes: Int = kFa3BlockN * head_dim * size_of[dtype]()
    comptime mbar_bytes: Int = 64
    comptime smem_bytes: Int = q_bytes + k_bytes + v_bytes + mbar_bytes

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

    # 3D TMA descriptors over the (B*L, H, D) gmem view. Tile = one
    # (BM/BN rows from a single head_idx, D cols).
    comptime gmem_shape = IndexList[3](
        UNKNOWN_VALUE, UNKNOWN_VALUE, head_dim
    )
    comptime q_smem_shape = IndexList[3](kFa3BlockM, 1, head_dim)
    comptime kv_smem_shape = IndexList[3](kFa3BlockN, 1, head_dim)

    var rows: Int = batch_int * seqlen_int
    var q_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, q_ptr, rows, nheads_int)
    var k_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, k_ptr, rows, nheads_int)
    var v_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, v_ptr, rows, nheads_int)

    comptime kernel_inst = fwd_fa3_kernel[
        dtype,
        head_dim,
        type_of(q_tma).tile_shape,
        type_of(q_tma).desc_shape,
        type_of(k_tma).tile_shape,
        type_of(k_tma).desc_shape,
    ]

    var compiled = ctx.compile_function[
        kernel_inst,
        kernel_inst,
    ](func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
        UInt32(smem_bytes)
    ))

    var grid = (ceildiv(seqlen_int, Int(kFa3BlockM)), nheads_int, batch_int)

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            q_tma,
            k_tma,
            v_tma,
            o_ptr,
            seqlen_int,
            nheads_int,
            softmax_scale,
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
            q_tma,
            k_tma,
            v_tma,
            o_ptr,
            seqlen_int,
            nheads_int,
            softmax_scale,
            o_b_stride,
            o_l_stride,
            o_h_stride,
            grid_dim=grid,
            block_dim=(kFa3NThreads,),
            shared_mem_bytes=smem_bytes,
        )

    comptime if not use_external_stream:
        ctx.synchronize()
