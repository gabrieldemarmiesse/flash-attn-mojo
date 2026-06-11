"""Static variant entry point for the FA4-target bwd kernels.

One compiled module exports the three launches (preprocess, main,
convert) plus acquire_ctx. Comptime params come from `-D` defines
passed by ``_jit.py``. Runtime arg layouts are documented in
``bwd_fa4/_jit.py``.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from launch import launch_bwd_preprocess, launch_bwd_main, launch_bwd_convert
from _ctx import acquire_ctx_handle

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.bfloat16]()
comptime HEAD_DIM: Int = get_defined_int["HEAD_DIM"]()
comptime USE_EXTERNAL_STREAM: Bool = get_defined_bool["USE_EXTERNAL_STREAM"]()
comptime CAUSAL: Bool = get_defined_bool["CAUSAL", False]()
comptime GQA_RATIO: Int = get_defined_int["GQA_RATIO", 1]()


def flash_attn_bwd_fa4_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def flash_attn_bwd_fa4_preprocess(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    launch_bwd_preprocess[DTYPE, HEAD_DIM, USE_EXTERNAL_STREAM, CAUSAL, GQA_RATIO](
        Int(py=args[0]),  # batch
        Int(py=args[1]),  # seqlen
        Int(py=args[2]),  # nheads
        Int(py=args[3]),  # o_addr
        Int(py=args[4]),  # do_addr
        Int(py=args[5]),  # lse_addr
        Int(py=args[6]),  # dpsum_addr
        Int(py=args[7]),  # lse_log2_addr
        Int(py=args[8]),  # dq_accum_addr
        Int(py=args[9]),  # dk_accum_addr (GQA; 0 otherwise)
        Int(py=args[10]),  # dv_accum_addr (GQA; 0 otherwise)
        Int(py=args[11]),  # stream
        Int(py=args[12]),  # ctx handle
    )
    return PythonObject(None)


def flash_attn_bwd_fa4_main(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    launch_bwd_main[DTYPE, HEAD_DIM, USE_EXTERNAL_STREAM, CAUSAL, GQA_RATIO](
        Int(py=args[0]),  # batch
        Int(py=args[1]),  # seqlen
        Int(py=args[2]),  # nheads
        Float32(py=args[3]),  # softmax_scale
        Int(py=args[4]),  # q_addr
        Int(py=args[5]),  # k_addr
        Int(py=args[6]),  # v_addr
        Int(py=args[7]),  # do_addr
        Int(py=args[8]),  # dk_addr
        Int(py=args[9]),  # dv_addr
        Int(py=args[10]),  # lse_log2_addr
        Int(py=args[11]),  # dpsum_addr
        Int(py=args[12]),  # dq_accum_addr
        Int(py=args[13]),  # stream
        Int(py=args[14]),  # ctx handle
    )
    return PythonObject(None)


def flash_attn_bwd_fa4_convert(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    launch_bwd_convert[DTYPE, HEAD_DIM, USE_EXTERNAL_STREAM, CAUSAL, GQA_RATIO](
        Int(py=args[0]),  # batch
        Int(py=args[1]),  # seqlen
        Int(py=args[2]),  # nheads
        Float32(py=args[3]),  # softmax_scale
        Int(py=args[4]),  # dq_accum_addr
        Int(py=args[5]),  # dq_addr
        Int(py=args[6]),  # stream
        Int(py=args[7]),  # ctx handle
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[flash_attn_bwd_fa4_preprocess](
            "flash_attn_bwd_fa4_preprocess"
        )
        m.def_py_function[flash_attn_bwd_fa4_main]("flash_attn_bwd_fa4_main")
        m.def_py_function[flash_attn_bwd_fa4_convert](
            "flash_attn_bwd_fa4_convert"
        )
        m.def_py_function[flash_attn_bwd_fa4_acquire_ctx](
            "flash_attn_bwd_fa4_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
