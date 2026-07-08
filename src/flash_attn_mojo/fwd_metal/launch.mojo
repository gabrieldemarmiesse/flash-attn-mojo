"""Launch helper for the Apple-GPU (Metal) forward attention kernel.

Unlike the CUDA path (which enqueues directly over torch's device
pointers — CUDA VA is process-global), Metal binds kernel arguments to
MTLBuffers the runtime owns, and torch's MPS allocator is a separate
registry mojo cannot bind. So this launcher receives HOST (CPU)
pointers, stages them through mojo-owned Metal device buffers, runs
the kernel, and copies the results back. On Apple unified memory the
staging copies are memory-bandwidth-bound (~2% at the canonical big
shape; the kernel dominates). See METAL_PLAN.md.

AIR dump: when the build defines `MOJO_DUMP_PTX=<path>` (wired from
the env var by `_jit.py`), the device function's textual AIR is written
to <path> at first-call JIT time via `compile_function(dump_asm=...)`.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host.device_context import (
    _DeviceContextPtr,
    _DeviceContextCpp,
    _DumpPath,
)
from std.math import ceildiv
from std.sys import get_defined_string

from kernel import fwd_metal_kernel
from common import BR, TPB, LOG2E

comptime MOJO_DUMP_PTX: StaticString = get_defined_string[
    "MOJO_DUMP_PTX", ""
]()


def _dump_path() -> _DumpPath:
    comptime if MOJO_DUMP_PTX == StaticString(""):
        return _DumpPath(False)
    else:
        return _DumpPath(MOJO_DUMP_PTX)


def launch_fwd_metal[
    head_dim: Int,
](
    batch: Int,
    seqlen: Int,
    nheads: Int,
    softmax_scale: Float32,
    q_host: Int,
    k_host: Int,
    v_host: Int,
    o_host: Int,
    lse_host: Int,
    ctx_handle_addr: Int,
) raises:
    var raw_ctx = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx))

    # (B, S, H, D) contiguous; the kernel indexes head-major within a
    # per-(b) slab. We fold batch into the head dimension: each batch
    # row is nheads independent (S, D) attention problems, so a single
    # flat (B*H, S, D) view works with head = b*H + h.
    comptime dtype = DType.float16
    var n = batch * seqlen * nheads * head_dim
    var lse_n = batch * nheads * seqlen

    var q_buf = ctx.enqueue_create_buffer[dtype](n)
    var k_buf = ctx.enqueue_create_buffer[dtype](n)
    var v_buf = ctx.enqueue_create_buffer[dtype](n)
    var o_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var l_buf = ctx.enqueue_create_buffer[DType.float32](lse_n)

    q_buf.enqueue_copy_from(
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=q_host)
    )
    k_buf.enqueue_copy_from(
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=k_host)
    )
    v_buf.enqueue_copy_from(
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=v_host)
    )

    # (B, S, H, D) row-major -> the kernel walks head-major slabs of
    # (S, D). With B*H flattened heads and per-head stride S*D, the
    # kernel's `head * seq * D` base is correct ONLY when the layout is
    # (B, H, S, D). The public wrapper hands us that transposed view
    # (q.transpose(1,2)), so here head in [0, B*H) maps to the right
    # (b, h) slab directly.
    var flat_heads = batch * nheads
    var scale_log2 = Float32(LOG2E) * softmax_scale
    var grid = ceildiv(seqlen, BR) * flat_heads

    var compiled = ctx.compile_function[
        fwd_metal_kernel[head_dim],
        dump_asm = _dump_path(),
    ]()
    ctx.enqueue_function(
        compiled,
        q_buf.unsafe_ptr(),
        k_buf.unsafe_ptr(),
        v_buf.unsafe_ptr(),
        o_buf.unsafe_ptr(),
        l_buf.unsafe_ptr(),
        seqlen,
        scale_log2,
        grid_dim=grid,
        block_dim=TPB,
    )

    o_buf.enqueue_copy_to(
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=o_host
        )
    )
    l_buf.enqueue_copy_to(
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=lse_host
        )
    )
    ctx.synchronize()
    _ = q_buf
    _ = k_buf
    _ = v_buf
    _ = o_buf
    _ = l_buf
