"""Static variant entry point for the FA4-target fwd kernel.

Comptime params come from `-D` defines passed by ``_jit.py``
(DTYPE, HEAD_DIM, USE_EXTERNAL_STREAM, and optionally MOJO_DUMP_PTX
which is consumed inside ``launch.mojo``). Runtime args are
documented in ``fwd_fa4/_jit.py::call_fwd_fa4``.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from launch import launch_fwd_fa4
from _ctx import acquire_ctx_handle

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.bfloat16]()
comptime HEAD_DIM: Int = get_defined_int["HEAD_DIM"]()
comptime USE_EXTERNAL_STREAM: Bool = get_defined_bool["USE_EXTERNAL_STREAM"]()


def flash_attn_fwd_fa4_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def flash_attn_fwd_fa4_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr: Int = Int(py=args[0])
    var k_addr: Int = Int(py=args[1])
    var v_addr: Int = Int(py=args[2])
    var o_addr: Int = Int(py=args[3])
    var lse_addr: Int = Int(py=args[4])
    var batch_int: Int = Int(py=args[5])
    var seqlen_int: Int = Int(py=args[6])
    var nheads_int: Int = Int(py=args[7])
    var softmax_scale: Float32 = Float32(py=args[8])
    var o_b_stride: Int = Int(py=args[9])
    var o_l_stride: Int = Int(py=args[10])
    var o_h_stride: Int = Int(py=args[11])
    var stream_handle_addr: Int = Int(py=args[12])
    # args[13..15] are comptime gates (read via get_defined_*).
    var ctx_handle_addr: Int = Int(py=args[16])

    if batch_int == 0 or seqlen_int == 0 or nheads_int == 0:
        return PythonObject(None)

    launch_fwd_fa4[
        DTYPE,
        HEAD_DIM,
        USE_EXTERNAL_STREAM,
    ](
        batch_int,
        seqlen_int,
        nheads_int,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        o_addr,
        lse_addr,
        o_b_stride,
        o_l_stride,
        o_h_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[flash_attn_fwd_fa4_variant](
            "flash_attn_fwd_fa4_variant"
        )
        m.def_py_function[flash_attn_fwd_fa4_acquire_ctx](
            "flash_attn_fwd_fa4_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
