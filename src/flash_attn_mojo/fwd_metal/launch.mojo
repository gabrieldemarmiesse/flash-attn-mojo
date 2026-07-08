"""Launch helper for the Apple-GPU (Metal) forward attention kernel.

Zero-copy over torch's own MPS buffers — the CUDA path's model, adapted
to Metal. The five pointer args are Metal GPU virtual addresses of
torch MPS tensors, extracted host-side via the `gpuAddress` Obj-C
selector (see `_mps.py`); we bind them straight as kernel arguments,
exactly like the CUDA launcher binds torch's CUDA VAs. No mojo-owned
staging buffers, no host round-trip. The Python wrapper
(`fwd_metal/__init__.py::fa_metal_fwd`) has already laid the inputs out
head-major fp16 on-device, revived the backing MTLHeaps, and flushed
torch's queue — so every buffer here is resident and torch's staging
writes have landed. `ctx.synchronize()` at the end flushes our queue so
torch's subsequent reads of O/LSE see the results (shared MTLDevice,
coherent once both queues flush).

AIR dump: when the build defines `MOJO_DUMP_PTX=<path>` (wired from
the env var by `_jit.py`), the device function's textual AIR is written
to <path> at first-call JIT time via `compile_function(dump_asm=...)`.
"""

from std.gpu.host import DeviceContext
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
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    o_addr: Int,
    lse_addr: Int,
    ctx_handle_addr: Int,
) raises:
    # Reconstruct a non-owning DeviceContext from the cached handle (see
    # `_ctx.mojo`) — avoids re-creating the context per call.
    var raw_ctx = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx))

    comptime dtype = DType.float16

    # The addresses are Metal GPU VAs of torch's MPS tensors. Wrap them
    # as plain device pointers — identical to how the current buffer path
    # passed `buf.unsafe_ptr()`, but the buffer is torch's, not ours. The
    # inputs are (B, H, S, D) contiguous fp16 (the public wrapper folds
    # batch into the head axis and expands GQA), so head in [0, B*H) maps
    # straight to its (S, D) slab via the kernel's `head * seq * D` base.
    # O is (B, H, S, D) fp32, LSE is (B, H, S) fp32.
    var q_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=q_addr
    )
    var k_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=k_addr
    )
    var v_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=v_addr
    )
    var o_ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=o_addr
    )
    var l_ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=lse_addr
    )

    var flat_heads = batch * nheads
    var scale_log2 = Float32(LOG2E) * softmax_scale
    # 1-D grid, q-tile-major within a head (see kernel.mojo): consecutive
    # threadgroup ids are q tiles of the SAME head.
    var grid = ceildiv(seqlen, BR) * flat_heads

    var compiled = ctx.compile_function[
        fwd_metal_kernel[head_dim],
        dump_asm = _dump_path(),
    ]()
    ctx.enqueue_function(
        compiled,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        seqlen,
        scale_log2,
        grid_dim=grid,
        block_dim=TPB,
    )
    ctx.synchronize()
