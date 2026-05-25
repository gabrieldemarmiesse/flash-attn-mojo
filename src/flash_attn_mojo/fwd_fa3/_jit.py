"""JIT-on-first-use dispatcher for the FA3 (sm_90+) forward kernel.

Same structure as ``fwd/_jit.py`` — the FA3 path is a parallel
subpackage selected at runtime when the GPU is sm_90+. The MVP
covers bf16 / head_dim=64 / non-causal / no-dropout only; other
configs fall back to the FA2 ``fwd`` path. The FA3 runtime tuple is
much smaller than FA2's (no causal/softcap/alibi/window/dropout/LSE
fields), so the index layout here is unrelated to ``fwd/_jit.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from flash_attn_mojo._jit_common import compile_and_load, detect_gpu_backend

_FWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _FWD_DIR.parent
_VARIANT_MOJO = _FWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_fwd_fa3(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single FA3 fwd call.

    ``args`` layout (see ``fwd_fa3/__init__.py::native_fwd_fa3``):
        0..3   q_addr, k_addr, v_addr, o_addr
        4..6   batch, seqlen, nheads
        7      softmax_scale
        8..10  q_b_stride, q_l_stride, q_h_stride
        11..13 k_*  (same triple)
        14..16 v_*
        17..19 o_*
        20     stream_handle_addr
        21     dtype_code     (comptime)
        22     head_dim       (comptime)
        23     use_external_stream (comptime)
    ``ctx_handle`` is appended as index 24 by this dispatcher.
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[21]
    head_dim = args[22]
    use_external_stream = bool(args[23])
    return (dtype_code, head_dim, use_external_stream)


def _mod_name(config: tuple) -> str:
    (dt, hd, ues) = config
    return f"{_DTYPE_NAME[dt]}_hd{hd}_extstr{int(ues)}"


def _defines(config: tuple) -> dict[str, str]:
    (dt, hd, ues) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "HEAD_DIM": str(hd),
        "USE_EXTERNAL_STREAM": b(ues),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="fwd_fa3",
        source_file=_VARIANT_MOJO,
        include_dirs=(_FWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.flash_attn_fwd_fa3_variant
    acquire = module.flash_attn_fwd_fa3_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
