"""Launch helpers for the FA4-target bwd kernels (preprocess, main,
convert). Scope (v1): bf16, head_dim=128, non-causal, contiguous
(B, L, H, D), seqlen % 128 == 0, Hq == Hk.

Side tensors (allocated by the Python wrapper):
  dpsum, lse_log2: (B, H, S) fp32 contiguous.
  dq_accum:        (B, H, S, D) fp32 contiguous (zeroed by preprocess).

PTX dump: `-D MOJO_DUMP_PTX=<path>` dumps the *main* kernel's PTX.
"""

from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.device_context import (
    _DeviceContextPtr,
    _DeviceContextCpp,
    _DumpPath,
)
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.math import ceildiv
from std.memory import OpaquePointer
from std.sys import get_defined_string, size_of
from std.utils.index import IndexList

from layout import UNKNOWN_VALUE
from layout.tma_async import create_split_tma

from kernel import bwd_main_kernel, bwd_preprocess_kernel, bwd_convert_kernel
from common import (
    kBwdBlockM,
    kBwdBlockN,
    kBwdNThreads,
    kBwdQdOStages,
    kBwdPreBlockM,
    kBwdPreThreads,
)

comptime MOJO_DUMP_PTX: StaticString = get_defined_string[
    "MOJO_DUMP_PTX", ""
]()


fn _dump_ptx_path() -> _DumpPath:
    comptime if MOJO_DUMP_PTX == StaticString(""):
        return _DumpPath(False)
    else:
        return _DumpPath(MOJO_DUMP_PTX)


fn _ctx_and_stream(
    ctx_handle_addr: Int,
) -> DeviceContext:
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutExternalOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    return DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))


def launch_bwd_preprocess[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    o_addr: Int,
    do_addr: Int,
    lse_addr: Int,
    dpsum_addr: Int,
    lse_log2_addr: Int,
    dq_accum_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    var ctx = _ctx_and_stream(ctx_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    var o_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=o_addr
    )
    var do_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=do_addr
    )
    var lse_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=lse_addr
    )
    var dpsum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dpsum_addr
    )
    var lse_log2_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=lse_log2_addr
    )
    var dq_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dq_accum_addr
    )

    comptime kernel_inst = bwd_preprocess_kernel[dtype, head_dim]
    var compiled = ctx.compile_function[kernel_inst, kernel_inst]()
    var grid = (
        ceildiv(seqlen_int, Int(kBwdPreBlockM)),
        nheads_int,
        batch_int,
    )

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            o_ptr,
            do_ptr,
            lse_ptr,
            dpsum_ptr,
            lse_log2_ptr,
            dq_accum_ptr,
            seqlen_int,
            nheads_int,
            grid_dim=grid,
            block_dim=(kBwdPreThreads,),
        )
    else:
        ctx.enqueue_function(
            compiled,
            o_ptr,
            do_ptr,
            lse_ptr,
            dpsum_ptr,
            lse_log2_ptr,
            dq_accum_ptr,
            seqlen_int,
            nheads_int,
            grid_dim=grid,
            block_dim=(kBwdPreThreads,),
        )
        ctx.synchronize()


def launch_bwd_main[
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
    do_addr: Int,
    dk_addr: Int,
    dv_addr: Int,
    lse_log2_addr: Int,
    dpsum_addr: Int,
    dq_accum_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    var ctx = _ctx_and_stream(ctx_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # Smem: K + V (BN x D) + 6-slot Q/dO ring (BM x D each) + 2 sdS
    # stages (BM x BN). hdim128: 32K + 32K + 96K + 32K = 192 KiB.
    comptime kv_bytes: Int = kBwdBlockN * head_dim * size_of[dtype]()
    comptime q_slot_bytes: Int = kBwdBlockM * head_dim * size_of[dtype]()
    comptime sds_bytes: Int = kBwdBlockM * kBwdBlockN * size_of[dtype]()
    comptime mbar_bytes: Int = 256
    # + 2-stage lse_log2/dpsum staging ring (2 x 2 x BM f32).
    comptime lse_dps_bytes: Int = 2 * (kBwdQdOStages // 2) * kBwdBlockM * 4
    comptime smem_bytes: Int = (
        2 * kv_bytes
        + kBwdQdOStages * q_slot_bytes
        + 2 * sds_bytes
        + lse_dps_bytes
        + mbar_bytes
    )

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
    var dk_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=dk_addr
    )
    var dv_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=dv_addr
    )
    var lse_log2_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=lse_log2_addr
    )
    var dpsum_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=dpsum_addr
    )
    var dq_accum_ptr = UnsafePointer[Float32, MutAnyOrigin](
        unsafe_from_address=dq_accum_addr
    )

    comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, head_dim)
    comptime q_smem_shape = IndexList[3](kBwdBlockM, 1, head_dim)
    comptime kv_smem_shape = IndexList[3](kBwdBlockN, 1, head_dim)

    var rows: Int = batch_int * seqlen_int
    var q_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, q_ptr, rows, nheads_int)
    var do_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, do_ptr, rows, nheads_int)
    var k_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, k_ptr, rows, nheads_int)
    var v_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode=swizzle
    ](ctx, v_ptr, rows, nheads_int)
    var dk_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
    ](ctx, dk_ptr, rows, nheads_int)
    var dv_tma = create_split_tma[
        kv_smem_shape, gmem_shape, swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
    ](ctx, dv_ptr, rows, nheads_int)

    comptime kernel_inst = bwd_main_kernel[
        dtype,
        head_dim,
        type_of(q_tma).tile_shape,
        type_of(q_tma).desc_shape,
        type_of(k_tma).tile_shape,
        type_of(k_tma).desc_shape,
        type_of(dk_tma).tile_shape,
        type_of(dk_tma).desc_shape,
    ]

    var compiled = ctx.compile_function[
        kernel_inst,
        kernel_inst,
        dump_asm = _dump_ptx_path(),
    ](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_bytes)
        )
    )

    var grid = (
        ceildiv(seqlen_int, Int(kBwdBlockN)),
        nheads_int,
        batch_int,
    )

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            q_tma,
            do_tma,
            k_tma,
            v_tma,
            dk_tma,
            dv_tma,
            lse_log2_ptr,
            dpsum_ptr,
            dq_accum_ptr,
            seqlen_int,
            softmax_scale,
            grid_dim=grid,
            block_dim=(kBwdNThreads,),
            shared_mem_bytes=smem_bytes,
        )
    else:
        ctx.enqueue_function(
            compiled,
            q_tma,
            do_tma,
            k_tma,
            v_tma,
            dk_tma,
            dv_tma,
            lse_log2_ptr,
            dpsum_ptr,
            dq_accum_ptr,
            seqlen_int,
            softmax_scale,
            grid_dim=grid,
            block_dim=(kBwdNThreads,),
            shared_mem_bytes=smem_bytes,
        )
        ctx.synchronize()


def launch_bwd_convert[
    dtype: DType,
    head_dim: Int,
    use_external_stream: Bool,
](
    batch_int: Int,
    seqlen_int: Int,
    nheads_int: Int,
    softmax_scale: Float32,
    dq_accum_addr: Int,
    dq_addr: Int,
    stream_handle_addr: Int,
    ctx_handle_addr: Int,
) raises:
    var ctx = _ctx_and_stream(ctx_handle_addr)
    var stream_opaque = OpaquePointer[MutAnyOrigin](
        unsafe_from_address=stream_handle_addr
    )

    var dq_accum_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=dq_accum_addr
    )
    var dq_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=dq_addr
    )

    # 128 x (128+4) f32 transpose tile.
    comptime cvt_smem_bytes: Int = head_dim * (kBwdPreBlockM + 4) * 4

    comptime kernel_inst = bwd_convert_kernel[dtype, head_dim]
    var compiled = ctx.compile_function[kernel_inst, kernel_inst](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(cvt_smem_bytes)
        )
    )
    var grid = (
        ceildiv(seqlen_int, Int(kBwdPreBlockM)),
        nheads_int,
        batch_int,
    )

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            dq_accum_ptr,
            dq_ptr,
            seqlen_int,
            nheads_int,
            softmax_scale,
            grid_dim=grid,
            block_dim=(kBwdPreThreads,),
            shared_mem_bytes=cvt_smem_bytes,
        )
    else:
        ctx.enqueue_function(
            compiled,
            dq_accum_ptr,
            dq_ptr,
            seqlen_int,
            nheads_int,
            softmax_scale,
            grid_dim=grid,
            block_dim=(kBwdPreThreads,),
            shared_mem_bytes=cvt_smem_bytes,
        )
        ctx.synchronize()
