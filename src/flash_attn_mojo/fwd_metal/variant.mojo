"""Static variant entry point for the Apple-GPU (Metal) fwd kernel.

Comptime params come from `-D` defines passed by ``_jit.py`` (DTYPE,
HEAD_DIM, and optionally MOJO_DUMP_PTX consumed in ``launch.mojo``).
Runtime args are documented in ``fwd_metal/_jit.py::call_fwd_metal``.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_int

from launch import launch_fwd_metal
from _ctx import acquire_ctx_handle

comptime HEAD_DIM: Int = get_defined_int["HEAD_DIM"]()


def flash_attn_fwd_metal_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def flash_attn_fwd_metal_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_host: Int = Int(py=args[0])
    var k_host: Int = Int(py=args[1])
    var v_host: Int = Int(py=args[2])
    var o_host: Int = Int(py=args[3])
    var lse_host: Int = Int(py=args[4])
    var batch: Int = Int(py=args[5])
    var seqlen: Int = Int(py=args[6])
    var nheads: Int = Int(py=args[7])
    var softmax_scale: Float32 = Float32(py=args[8])
    # args[9] is the dtype code, args[10] head_dim — comptime gates.
    var ctx_handle_addr: Int = Int(py=args[11])

    if batch == 0 or seqlen == 0 or nheads == 0:
        return PythonObject(None)

    launch_fwd_metal[HEAD_DIM](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_host,
        k_host,
        v_host,
        o_host,
        lse_host,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[flash_attn_fwd_metal_variant](
            "flash_attn_fwd_metal_variant"
        )
        m.def_py_function[flash_attn_fwd_metal_acquire_ctx](
            "flash_attn_fwd_metal_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
