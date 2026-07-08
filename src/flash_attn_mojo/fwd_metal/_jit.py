"""JIT-on-first-use dispatcher for the Apple-GPU (Metal) fwd kernel.

Same structure as the other subpackages' ``_jit.py``. The comptime
config is just (dtype, head_dim) plus the optional ``MOJO_DUMP_PTX``
env var (forwarded as a ``-D`` define so a dump build gets its own
cache slot).
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

_DTYPE_NAME = {0: "fp16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 2: "float32"}


def call_fwd_metal(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single fwd call.

    ``args`` layout (see ``fwd_metal/__init__.py::native_fwd_metal``):
        0..4   q_host, k_host, v_host, o_host, lse_host  (CPU ptrs)
        5..7   batch, seqlen, nheads
        8      softmax_scale
        9      dtype_code  (comptime)
        10     head_dim    (comptime)
    ``ctx_handle`` is appended as index 11 by this dispatcher.
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[9]
    head_dim = args[10]
    dump_ptx = os.environ.get("MOJO_DUMP_PTX", "")
    return (dtype_code, head_dim, dump_ptx)


def _mod_name(config: tuple) -> str:
    dt, hd, dump_ptx = config
    suffix = "_dumpptx" if dump_ptx else ""
    return f"{_DTYPE_NAME[dt]}_hd{hd}{suffix}"


def _defines(config: tuple) -> dict[str, str]:
    dt, hd, dump_ptx = config
    defines = {"DTYPE": _DTYPE_DEFINE[dt], "HEAD_DIM": str(hd)}
    if dump_ptx:
        defines["MOJO_DUMP_PTX"] = dump_ptx
    return defines


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="fwd_metal",
        source_file=_VARIANT_MOJO,
        include_dirs=(_FWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.flash_attn_fwd_metal_variant
    acquire = module.flash_attn_fwd_metal_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
