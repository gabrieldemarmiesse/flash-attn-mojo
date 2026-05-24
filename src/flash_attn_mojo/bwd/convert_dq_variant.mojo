"""Static variant entry point for bwd convert-dQ.

Mirrors `bwd/preprocess_variant.mojo`: comptime params read from `-D`
defines so a single source file covers every config.

Runtime args tuple is built in `bwd/__init__.py::native_bwd_convert_dq`.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from convert_dq_launch import launch_bwd_convert_dq
from _ctx import acquire_ctx_handle

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.bfloat16]()
comptime HEAD_DIM: Int = get_defined_int["HEAD_DIM"]()
comptime USE_EXTERNAL_STREAM: Bool = get_defined_bool["USE_EXTERNAL_STREAM"]()


def bwd_convert_dq_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def bwd_convert_dq_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var dqaccum_addr: Int = Int(py=args[0])
    var dq_addr: Int = Int(py=args[1])
    var batch_int: Int = Int(py=args[2])
    var seqlen_int: Int = Int(py=args[3])
    var nheads_int: Int = Int(py=args[4])
    var dqaccum_b_stride: Int = Int(py=args[5])
    var dqaccum_h_stride: Int = Int(py=args[6])
    var dqaccum_l_stride: Int = Int(py=args[7])
    var dq_b_stride: Int = Int(py=args[8])
    var dq_l_stride: Int = Int(py=args[9])
    var dq_h_stride: Int = Int(py=args[10])
    var stream_handle_addr: Int = Int(py=args[11])
    # args[12..14] are comptime defines (dtype, head_dim, use_ext_stream);
    # skipped here, read at module level via get_defined_*. ctx_handle is
    # appended by `call_bwd_convert_dq` as the 16th positional (index 15).
    var ctx_handle_addr: Int = Int(py=args[15])

    if batch_int == 0 or seqlen_int == 0 or nheads_int == 0:
        return PythonObject(None)

    launch_bwd_convert_dq[DTYPE, HEAD_DIM, USE_EXTERNAL_STREAM](
        batch_int,
        seqlen_int,
        nheads_int,
        dqaccum_addr,
        dq_addr,
        dqaccum_b_stride,
        dqaccum_h_stride,
        dqaccum_l_stride,
        dq_b_stride,
        dq_l_stride,
        dq_h_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[bwd_convert_dq_variant]("bwd_convert_dq_variant")
        m.def_py_function[bwd_convert_dq_acquire_ctx](
            "bwd_convert_dq_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
