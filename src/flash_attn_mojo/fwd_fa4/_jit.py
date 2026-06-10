"""JIT-on-first-use dispatcher for the FA4-target fwd kernel.

Same structure as the other subpackages' ``_jit.py``. One extra
knob: the ``MOJO_DUMP_PTX`` environment variable. When set, its
value is forwarded as a ``-D MOJO_DUMP_PTX=<path>`` define;
``launch.mojo`` reads it via ``get_defined_string`` and passes it to
``compile_function(dump_asm=...)``, so the device PTX is written to
that path when the kernel JITs on first call. The define
participates in the ``.so`` content hash, so dump builds get their
own cache slot and a plain rerun with the variable set still
recompiles + dumps.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from flash_attn_mojo._jit_common import compile_and_load, detect_gpu_backend

_FWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _FWD_DIR.parent
_VARIANT_MOJO = _FWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_fwd_fa4(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single fwd call.

    ``args`` layout (see ``fwd_fa4/__init__.py::native_fwd_fa4``):
        0..3   q_addr, k_addr, v_addr, o_addr
        4..6   batch, seqlen, nheads
        7      softmax_scale
        8..10  o_b_stride, o_l_stride, o_h_stride
        11     stream_handle_addr
        12     dtype_code     (comptime)
        13     head_dim       (comptime)
        14     use_external_stream (comptime)
    ``ctx_handle`` is appended as index 15 by this dispatcher.
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[12]
    head_dim = args[13]
    use_external_stream = bool(args[14])
    dump_ptx = os.environ.get("MOJO_DUMP_PTX", "")
    return (dtype_code, head_dim, use_external_stream, dump_ptx)


def _mod_name(config: tuple) -> str:
    (dt, hd, ues, dump_ptx) = config
    suffix = "_dumpptx" if dump_ptx else ""
    return f"{_DTYPE_NAME[dt]}_hd{hd}_extstr{int(ues)}{suffix}"


def _defines(config: tuple) -> dict[str, str]:
    (dt, hd, ues, dump_ptx) = config

    defines = {
        "DTYPE": _DTYPE_DEFINE[dt],
        "HEAD_DIM": str(hd),
        "USE_EXTERNAL_STREAM": "true" if ues else "false",
    }
    if dump_ptx:
        defines["MOJO_DUMP_PTX"] = dump_ptx
    return defines


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="fwd_fa4",
        source_file=_VARIANT_MOJO,
        include_dirs=(_FWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.flash_attn_fwd_fa4_variant
    acquire = module.flash_attn_fwd_fa4_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
