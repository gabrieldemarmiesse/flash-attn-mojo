"""Apple Silicon (MPS) interop helpers for the zero-copy Metal bridge.

The Mojo GPU kernels run on Apple Metal via `DeviceContext`. A Mojo
`DeviceBuffer`'s internal `_device_ptr` is the Metal-3.1 `gpuAddress`
of the underlying `MTLBuffer` (a 64-bit GPU virtual address). On
NVIDIA, torch's `tensor.data_ptr()` is already a CUDA device VA, so
plumbing it straight through to the kernel works. On Apple, however,
torch's `tensor.data_ptr()` is the `id<MTLBuffer>` Obj-C object
pointer (verified: `object_getClassName` -> `AGXG16GFamilyBuffer`),
NOT the GPU VA — Mojo can't dereference it.

We extract the GPU VA via the same selector Mojo uses internally:

  [storage.MTLBuffer gpuAddress] + (tensor.data_ptr() - storage.data_ptr())

The base `gpuAddress` is per-`MTLBuffer`; the tensor's storage offset
(stored by torch as a byte delta on top of the buffer obj pointer)
gives the per-tensor location. Result: a 64-bit GPU VA that Mojo's
JIT-compiled Metal kernel can use exactly like a CUDA pointer, with
no copy through host memory.

This is the piece `METAL_PLAN.md` (line ~271, "zero-copy verified
impossible") was missing: its two negative tests both bound
`tensor.data_ptr()` — the Obj-C object, not the VA — and lacked heap
revival (below), so the kernel dereferenced garbage / wrote nowhere.
The sibling `causal-conv1d-mojo` repo runs this exact path in CI.

Synchronization: torch's MPS runs on its own command queue; Mojo's
`DeviceContext` runs on its own. The `MTLDevice` is shared, so memory
is consistent once both queues have flushed. The Python wrapper in
`fwd_metal/__init__.py` calls `revive_heaps(...)` + `torch.mps.synchronize()`
immediately before launching (flushing torch's staging writes), and
relies on `ctx.synchronize()` inside the JIT'd Mojo entry point to
flush our side before returning.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _objc() -> ctypes.CDLL:
    """Lazy-load libobjc and pre-bind the C ABI types for the few
    selectors we use. `objc_msgSend.argtypes` MUST be set on Apple
    Silicon — without it the ABI defaults are wrong for variadic
    msgSend and you'll get a segfault on entry.
    """
    libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    libobjc.sel_registerName.restype = ctypes.c_void_p
    libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
    libobjc.objc_msgSend.restype = ctypes.c_uint64
    libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return libobjc


@lru_cache(maxsize=1)
def _sel_gpu_address() -> int:
    libobjc = _objc()
    return libobjc.sel_registerName(b"gpuAddress")


def gpu_address(t: torch.Tensor) -> int:
    """Return the Metal GPU virtual address of `t`'s first element.

    Handles non-zero storage offsets (sliced views, etc.) by adding
    the byte delta between `tensor.data_ptr()` and the storage's
    `data_ptr()` to the buffer's base `gpuAddress`. The kernel reads
    elements via `base + head*seq*D + row*D + col`, with the offset
    already in bytes here, so this base + offset is enough.
    """
    storage = t.untyped_storage()
    buf_obj = storage.data_ptr()
    if buf_obj == 0:
        return 0
    libobjc = _objc()
    base_gpu = libobjc.objc_msgSend(buf_obj, _sel_gpu_address())
    offset_bytes = t.data_ptr() - buf_obj
    return base_gpu + offset_bytes


def gpu_address_or_zero(t: torch.Tensor | None) -> int:
    """Same as `gpu_address` but returns 0 for `None`."""
    if t is None:
        return 0
    return gpu_address(t)


# Revival cadence. macOS evicts idle GPU memory after ~1-1.5 s (measured
# on M4; matches ggml-org/llama.cpp#10119), so touching each heap at
# least every _REVIVE_WINDOW_S keeps it comfortably resident while making
# steady-state loops pay (almost) nothing.
# FLASH_ATTN_MOJO_MPS_REVIVE=always -> revive on every call (paranoid)
# FLASH_ATTN_MOJO_MPS_REVIVE=off    -> never revive (debugging only)
_REVIVE_MODE = os.environ.get("FLASH_ATTN_MOJO_MPS_REVIVE", "auto")
_REVIVE_WINDOW_S = 0.35
# Per-tensor last-revival stamps, keyed on `t.data_ptr()` (buffer object
# pointer + storage offset — the fast C-level accessor; `untyped_storage()`
# costs ~15 us per call in Python, data_ptr ~1 us). Keying per pointer
# (not one global stamp) matters: a call inside the window may still
# introduce a tensor whose heap has been idle for minutes (say, a second
# model's weights) — that tensor must force a revival even though the
# previous call's tensors are all still hot. Distinct views of one storage
# get distinct keys (harmless: each is revived once, then hot). Pointer
# reuse by torch's allocator is also harmless: a recycled MTLBuffer belongs
# to the same heap that was just touched through its previous owner.
_revive_stamp: dict[int, float] = {}


def revive_heaps(*tensors: torch.Tensor | None) -> None:
    """Touch each tensor with a tiny GPU op so its MTLHeap is resident
    when the Mojo kernel dispatches. Call this (followed by
    `torch.mps.synchronize()`) immediately before every Mojo launch.

    Why: torch MPS tensors are sub-allocations of hazard-tracked
    `MTLHeap`s, and macOS evicts idle GPU memory after ~1-1.5 s. Mojo's
    Metal backend only declares resources it allocated itself to its
    compute encoder (`useResource:` is skipped for foreign addresses),
    so a Mojo kernel referencing an evicted heap silently reads zeros
    and drops writes — no error anywhere. Any submitted torch op that
    references the buffer makes the driver re-map its heap, putting the
    immediately-following Mojo dispatch back inside the residency
    window. Heaps are revived individually (torch pools small and large
    allocations on different heaps), hence one touch per tensor.

    Cost control: the touches are batched into one `torch.stack` per
    dtype group (a single kernel reading one element of every tensor in
    the group — layout-agnostic 0-d views, no version-counter bumps),
    and skipped entirely when every argument's storage was revived
    < _REVIVE_WINDOW_S ago — far inside the driver's ~1-1.5 s eviction
    horizon, so a steady decode/bench loop revives at most ~3x per
    second instead of per call. First-call revival is load-bearing: the
    JIT compile can take seconds, during which freshly-staged heaps
    would otherwise re-idle — hence this runs *after* compile, right
    before dispatch.
    """
    if _REVIVE_MODE == "off":
        return
    live = [t for t in tensors if t is not None and t.numel() > 0]
    now = time.monotonic()
    if _REVIVE_MODE != "always":
        stamps = _revive_stamp
        if all(
            now - stamps.get(t.data_ptr(), 0.0) < _REVIVE_WINDOW_S
            for t in live
        ):
            return
    groups: dict[torch.dtype, list[torch.Tensor]] = {}
    for t in live:
        groups.setdefault(t.dtype, []).append(t[(0,) * t.dim()])
    for group in groups.values():
        torch.stack(group)
    if len(_revive_stamp) > 65536:
        _revive_stamp.clear()
    for t in live:
        _revive_stamp[t.data_ptr()] = now
