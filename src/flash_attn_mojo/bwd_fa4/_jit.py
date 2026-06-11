"""JIT-on-first-use dispatcher for the FA4-target bwd kernels.

One compiled `.so` (single cache entry) exports the three launches:
preprocess, main, convert. ``MOJO_DUMP_PTX`` is forwarded as a
``-D`` define and dumps the *main* kernel's PTX at JIT time (see
``fwd_fa4/_jit.py`` for the mechanism).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from flash_attn_mojo._jit_common import compile_and_load, detect_gpu_backend

_BWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _BWD_DIR.parent
_VARIANT_MOJO = _BWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_bwd_fa4_preprocess(args: tuple, config: tuple) -> None:
    """args: (batch, seqlen, nheads, o_addr, do_addr, lse_addr,
    dpsum_addr, lse_log2_addr, dq_accum_addr, stream)."""
    fns, ctx_handle = _get_variant_fns(config)
    fns[0](*args, ctx_handle)


def call_bwd_fa4_main(args: tuple, config: tuple) -> None:
    """args: (batch, seqlen, nheads, softmax_scale, q, k, v, do, dk,
    dv, lse_log2, dpsum, dq_accum addrs, stream)."""
    fns, ctx_handle = _get_variant_fns(config)
    fns[1](*args, ctx_handle)


def call_bwd_fa4_convert(args: tuple, config: tuple) -> None:
    """args: (batch, seqlen, nheads, softmax_scale, dq_accum_addr,
    dq_addr, stream)."""
    fns, ctx_handle = _get_variant_fns(config)
    fns[2](*args, ctx_handle)


def make_config(
    dtype_code: int,
    head_dim: int,
    use_external_stream: bool,
    causal: bool = False,
    gqa_ratio: int = 1,
):
    dump_ptx = os.environ.get("MOJO_DUMP_PTX", "")
    return (
        dtype_code, head_dim, bool(use_external_stream), bool(causal),
        int(gqa_ratio), dump_ptx,
    )


def _mod_name(config: tuple) -> str:
    (dt, hd, ues, causal, ratio, dump_ptx) = config
    suffix = "_causal" if causal else ""
    suffix += f"_gqa{ratio}" if ratio > 1 else ""
    suffix += "_dumpptx" if dump_ptx else ""
    return f"{_DTYPE_NAME[dt]}_hd{hd}_extstr{int(ues)}{suffix}"


def _defines(config: tuple) -> dict[str, str]:
    (dt, hd, ues, causal, ratio, dump_ptx) = config
    defines = {
        "DTYPE": _DTYPE_DEFINE[dt],
        "HEAD_DIM": str(hd),
        "USE_EXTERNAL_STREAM": "true" if ues else "false",
        "CAUSAL": "true" if causal else "false",
        "GQA_RATIO": str(ratio),
    }
    if dump_ptx:
        defines["MOJO_DUMP_PTX"] = dump_ptx
    return defines


@lru_cache(maxsize=None)
def _get_variant_fns(config: tuple):
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="bwd_fa4",
        source_file=_VARIANT_MOJO,
        include_dirs=(_BWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fns = (
        module.flash_attn_bwd_fa4_preprocess,
        module.flash_attn_bwd_fa4_main,
        module.flash_attn_bwd_fa4_convert,
    )
    ctx_handle = int(module.flash_attn_bwd_fa4_acquire_ctx(()))
    return fns, ctx_handle
