"""Static variant entry point for the main bwd kernel.

Mirrors `fwd/variant.mojo`: comptime params read from `-D` defines so
a single source file covers every (dtype × head_dim × external-stream)
config.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from launch import launch_bwd
from _ctx import acquire_ctx_handle

comptime DTYPE: DType = get_defined_dtype["DTYPE", DType.bfloat16]()
comptime HEAD_DIM: Int = get_defined_int["HEAD_DIM"]()
comptime CAUSAL: Bool = get_defined_bool["CAUSAL"]()
comptime USE_EXTERNAL_STREAM: Bool = get_defined_bool["USE_EXTERNAL_STREAM"]()


def bwd_main_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def bwd_main_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr: Int = Int(py=args[0])
    var k_addr: Int = Int(py=args[1])
    var v_addr: Int = Int(py=args[2])
    var do_addr: Int = Int(py=args[3])
    var lse_addr: Int = Int(py=args[4])
    var delta_addr: Int = Int(py=args[5])
    var dk_addr: Int = Int(py=args[6])
    var dv_addr: Int = Int(py=args[7])
    var dqa_addr: Int = Int(py=args[8])
    var batch_int: Int = Int(py=args[9])
    var seqlen_int: Int = Int(py=args[10])
    var nheads_q_int: Int = Int(py=args[11])
    var nheads_kv_int: Int = Int(py=args[12])
    var softmax_scale: Float32 = Float32(py=args[13])
    var q_b_stride: Int = Int(py=args[14])
    var q_l_stride: Int = Int(py=args[15])
    var q_h_stride: Int = Int(py=args[16])
    var k_b_stride: Int = Int(py=args[17])
    var k_l_stride: Int = Int(py=args[18])
    var k_h_stride: Int = Int(py=args[19])
    var v_b_stride: Int = Int(py=args[20])
    var v_l_stride: Int = Int(py=args[21])
    var v_h_stride: Int = Int(py=args[22])
    var do_b_stride: Int = Int(py=args[23])
    var do_l_stride: Int = Int(py=args[24])
    var do_h_stride: Int = Int(py=args[25])
    var dk_b_stride: Int = Int(py=args[26])
    var dk_l_stride: Int = Int(py=args[27])
    var dk_h_stride: Int = Int(py=args[28])
    var dv_b_stride: Int = Int(py=args[29])
    var dv_l_stride: Int = Int(py=args[30])
    var dv_h_stride: Int = Int(py=args[31])
    var lse_b_stride: Int = Int(py=args[32])
    var lse_h_stride: Int = Int(py=args[33])
    var delta_b_stride: Int = Int(py=args[34])
    var delta_h_stride: Int = Int(py=args[35])
    var dqa_b_stride: Int = Int(py=args[36])
    var dqa_h_stride: Int = Int(py=args[37])
    var dqa_l_stride: Int = Int(py=args[38])
    var stream_handle_addr: Int = Int(py=args[39])
    # args[40..43] are comptime defines (dtype, head_dim, causal,
    # use_ext_stream); ctx_handle is appended by `call_bwd_main` as the
    # 45th positional (index 44).
    var ctx_handle_addr: Int = Int(py=args[44])

    if batch_int == 0 or seqlen_int == 0 or nheads_q_int == 0 or nheads_kv_int == 0:
        return PythonObject(None)

    launch_bwd[DTYPE, HEAD_DIM, CAUSAL, USE_EXTERNAL_STREAM](
        batch_int,
        seqlen_int,
        nheads_q_int,
        nheads_kv_int,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        do_addr,
        lse_addr,
        delta_addr,
        dk_addr,
        dv_addr,
        dqa_addr,
        q_b_stride,
        q_l_stride,
        q_h_stride,
        k_b_stride,
        k_l_stride,
        k_h_stride,
        v_b_stride,
        v_l_stride,
        v_h_stride,
        do_b_stride,
        do_l_stride,
        do_h_stride,
        dk_b_stride,
        dk_l_stride,
        dk_h_stride,
        dv_b_stride,
        dv_l_stride,
        dv_h_stride,
        lse_b_stride,
        lse_h_stride,
        delta_b_stride,
        delta_h_stride,
        dqa_b_stride,
        dqa_h_stride,
        dqa_l_stride,
        stream_handle_addr,
        ctx_handle_addr,
    )
    return PythonObject(None)


@export
def PyInit_variant() -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[bwd_main_variant]("bwd_main_variant")
        m.def_py_function[bwd_main_acquire_ctx]("bwd_main_acquire_ctx")
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
