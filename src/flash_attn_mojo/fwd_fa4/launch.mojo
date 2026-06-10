"""Launch helper for the FA4-target fwd kernel.

Scope (v1): bf16, head_dim=128, non-causal, contiguous (B, L, H, D),
seqlen % BN == 0, Hq == Hk.

PTX dump: when the build defines `MOJO_DUMP_PTX=<path>` (wired from
the same-named environment variable by `_jit.py`), the device
function's PTX is written to <path> at first-call JIT time via
`compile_function(dump_asm=...)`. A `%` in the path expands to the
kernel module name.
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

from kernel import fwd_fa4_kernel
from common import kFa4NThreads, kFa4BlockM, kFa4BlockN, kFa4KVStages

comptime MOJO_DUMP_PTX: StaticString = get_defined_string[
    "MOJO_DUMP_PTX", ""
]()


fn _dump_ptx_path() -> _DumpPath:
    comptime if MOJO_DUMP_PTX == StaticString(""):
        return _DumpPath(False)
    else:
        return _DumpPath(MOJO_DUMP_PTX)


def launch_fwd_fa4[
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

    comptime swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B

    # Smem: Q (BM x D) + kFa4KVStages ring slots (BN x D) bf16 +
    # mbarriers. head_dim=128: 32 KiB per tile -> 224 KiB (H100
    # opt-in cap is 227 KiB).
    comptime q_bytes: Int = kFa4BlockM * head_dim * size_of[dtype]()
    comptime kv_slot_bytes: Int = kFa4BlockN * head_dim * size_of[dtype]()
    comptime mbar_bytes: Int = 128
    comptime smem_bytes: Int = (
        q_bytes + kFa4KVStages * kv_slot_bytes + mbar_bytes
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
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=o_addr
    )

    # 3D TMA descriptors over the (B*L, H, D) gmem view.
    comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, head_dim)
    comptime q_smem_shape = IndexList[3](kFa4BlockM, 1, head_dim)
    comptime kv_smem_shape = IndexList[3](kFa4BlockN, 1, head_dim)

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
    # O store descriptor: unswizzled, so the kernel can stage O in a
    # plain row-major smem tile (one whole-tile bulk store).
    var o_imm_ptr = UnsafePointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=o_addr
    )
    var o_tma = create_split_tma[
        q_smem_shape, gmem_shape, swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
    ](ctx, o_imm_ptr, rows, nheads_int)

    comptime kernel_inst = fwd_fa4_kernel[
        dtype,
        head_dim,
        type_of(q_tma).tile_shape,
        type_of(q_tma).desc_shape,
        type_of(k_tma).tile_shape,
        type_of(k_tma).desc_shape,
        type_of(o_tma).tile_shape,
        type_of(o_tma).desc_shape,
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

    var grid = (ceildiv(seqlen_int, Int(kFa4BlockM)), nheads_int, batch_int)

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            compiled,
            q_tma,
            k_tma,
            v_tma,
            o_tma,
            seqlen_int,
            softmax_scale,
            grid_dim=grid,
            block_dim=(kFa4NThreads,),
            shared_mem_bytes=smem_bytes,
        )
    else:
        ctx.enqueue_function(
            compiled,
            q_tma,
            k_tma,
            v_tma,
            o_tma,
            seqlen_int,
            softmax_scale,
            grid_dim=grid,
            block_dim=(kFa4NThreads,),
            shared_mem_bytes=smem_bytes,
        )
        ctx.synchronize()
