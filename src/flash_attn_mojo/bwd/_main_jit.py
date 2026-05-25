"""JIT-on-first-use dispatcher for the main bwd kernel.

Each unique runtime config (dtype × head_dim × use_external_stream)
compiles ``bwd/variant.mojo`` once via ``mojo build -D KEY=VALUE …`` and
caches the resulting ``.so`` on disk. Mirrors `fwd/_jit.py` and
`bwd/_preprocess_jit.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from flash_attn_mojo._jit_common import compile_and_load, detect_gpu_backend

_BWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _BWD_DIR.parent
_VARIANT_MOJO = _BWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_bwd_main(args: tuple) -> None:
    """JIT-compile (if needed) and dispatch a single bwd-main call.

    ``args`` is the runtime tuple built by
    ``bwd/__init__.py::native_bwd_main``. The comptime gating fields
    live at indices 49..52 (dtype, head_dim, causal, use_external_stream).
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    # Append ctx_handle as the next positional (index 53).
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[50]
    head_dim = args[51]
    causal = bool(args[52])
    use_external_stream = bool(args[53])
    return (dtype_code, head_dim, causal, use_external_stream)


def _mod_name(config: tuple) -> str:
    (dt, hd, causal, ues) = config
    causal_tag = "causal" if causal else "noncausal"
    return f"{_DTYPE_NAME[dt]}_hd{hd}_{causal_tag}_extstr{int(ues)}"


def _defines(config: tuple) -> dict[str, str]:
    (dt, hd, causal, ues) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "HEAD_DIM": str(hd),
        "CAUSAL": b(causal),
        "USE_EXTERNAL_STREAM": b(ues),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="bwd_main",
        source_file=_VARIANT_MOJO,
        include_dirs=(_BWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.bwd_main_variant
    acquire = module.bwd_main_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
