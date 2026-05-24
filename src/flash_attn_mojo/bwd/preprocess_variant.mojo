"""Static variant entry point for bwd_preprocess.

Mirrors `fwd/variant.mojo`: comptime params read from `-D` defines so
a single source file covers every config.

Runtime args tuple is built in `bwd/__init__.py::native_bwd_preprocess`.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from preprocess_launch import launch_bwd_preprocess
from _ctx import acquire_ctx_handle

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.bfloat16]()
comptime HEAD_DIM: Int = get_defined_int["HEAD_DIM"]()
comptime USE_EXTERNAL_STREAM: Bool = get_defined_bool["USE_EXTERNAL_STREAM"]()


def bwd_preprocess_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def bwd_preprocess_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var dout_addr: Int = Int(py=args[0])
    var o_addr: Int = Int(py=args[1])
    var delta_addr: Int = Int(py=args[2])
    var batch_int: Int = Int(py=args[3])
    var seqlen_int: Int = Int(py=args[4])
    var nheads_int: Int = Int(py=args[5])
    var do_b_stride: Int = Int(py=args[6])
    var do_l_stride: Int = Int(py=args[7])
    var do_h_stride: Int = Int(py=args[8])
    var o_b_stride: Int = Int(py=args[9])
    var o_l_stride: Int = Int(py=args[10])
    var o_h_stride: Int = Int(py=args[11])
    var delta_b_stride: Int = Int(py=args[12])
    var delta_h_stride: Int = Int(py=args[13])
    var stream_handle_addr: Int = Int(py=args[14])
    # args[15..17] are comptime defines (dtype, head_dim, use_ext_stream);
    # skipped here, read at module level via get_defined_*. ctx_handle is
    # appended by `call_bwd_preprocess` as the 19th positional (index 18).
    var ctx_handle_addr: Int = Int(py=args[18])

    if batch_int == 0 or seqlen_int == 0 or nheads_int == 0:
        return PythonObject(None)

    launch_bwd_preprocess[DTYPE, HEAD_DIM, USE_EXTERNAL_STREAM](
        batch_int,
        seqlen_int,
        nheads_int,
        dout_addr,
        o_addr,
        delta_addr,
        do_b_stride,
        do_l_stride,
        do_h_stride,
        o_b_stride,
        o_l_stride,
        o_h_stride,
        delta_b_stride,
        delta_h_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[bwd_preprocess_variant]("bwd_preprocess_variant")
        m.def_py_function[bwd_preprocess_acquire_ctx](
            "bwd_preprocess_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
