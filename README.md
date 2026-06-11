# flash-attn-mojo

FlashAttention-4-class attention kernels written from scratch in
[Mojo](https://www.modular.com/mojo), racing Tri Dao's
[FlashAttention-4](https://github.com/Dao-AILab/flash-attention)
(`flash_attn.cute`) on NVIDIA Hopper.

**Status: at parity.** At the canonical shape (B=2, S=8192, H=16,
D=128, bf16, H100) both the forward and the backward kernels match
FA4's kernel time within run-to-run variance:

| kernel       | ratio (mojo/FA4)            |
|--------------|-----------------------------|
| fwd          | 0.99x (mojo faster)         |
| bwd          | 1.00–1.06x                  |
| fwd (causal) | 0.99x (mojo faster)         |
| bwd (causal) | 0.99–1.05x                  |
| fwd (GQA 4x) | 1.01 / 0.99x (plain/causal) |
| bwd (GQA 4x) | 0.98 / 1.01x (plain/causal) |
| fwd (varlen) | 1.00 / 0.98x (plain/causal) |
| bwd (varlen) | 0.98 / 1.00x (plain/causal) |

(Locked clocks, interleaved kernel-only CUPTI timing; both kernels
wobble ~2–4% run to run, so everything above straddles parity. The
bwd main loop executes 448 instructions/iteration to FA4's 532 at
identical tensor-core work. Varlen rows: the canonical packed
config — 8 sequences, 16384 total tokens, lengths 1280–3072.)

The kernels are warp-specialized TMA + WGMMA Hopper kernels JIT-built
via `mojo build` on first use and cached; correctness sits at the
bf16 noise floor against an fp32 SDPA reference, and LSE matches to
fp32 precision.

## Supported envelope

`flash_attn_func(q, k, v, softmax_scale=None, causal=False, *,
return_lse=False)` with:

- bf16, contiguous `(batch, seqlen, nheads, head_dim)`
- `head_dim == 128`, `seqlen % 128 == 0`, MHA or GQA
  (`Hq % Hkv == 0`)
- causal or non-causal (both differentiable, both at parity); no
  dropout / windows / ALiBi
- CUDA sm90 (Hopper); non-CUDA tensors run a pure-PyTorch reference

Everything outside the envelope raises a clear error. The op is
differentiable (`out.backward(...)` runs the FA4-class backward:
preprocess → main → dq-convert, mirroring FA4's pipeline).

`flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)`
runs packed variable-length attention over `(total_tokens, nheads,
head_dim)` tensors (FA4's varlen layout, packed `(nheads, total_q)`
LSE), differentiable end-to-end, with ARBITRARY sequence lengths
(>= 1 — ragged tails are masked in-kernel and tail tiles stored
row-predicated). Current varlen envelope: self-attention lengths
(`cu_seqlens_q == cu_seqlens_k`); the backward additionally requires
MHA.

## Install

```bash
pip install flash-attn-mojo
```

For drop-in `import flash_attn` compatibility with code written
against upstream Tri Dao (within the envelope above):

```bash
pip install flash-attn-mojo-compatibility
```

(See `compat/README.md` — that package conflicts with the upstream
`flash-attn` package on the `flash_attn` import name.)

## Usage

```python
import torch
from flash_attn_mojo import flash_attn_func

q = torch.randn(2, 8192, 16, 128, dtype=torch.bfloat16,
                device="cuda", requires_grad=True)
k, v = torch.randn_like(q), torch.randn_like(q)

out = flash_attn_func(q, k, v)
out.backward(torch.randn_like(out))
```

The first call per config JIT-compiles (~1 s) and caches under
`$XDG_CACHE_HOME/flash_attn_mojo/`.

## Layout

- `src/flash_attn_mojo/` — main package (`fwd_fa4/`, `bwd_fa4/`
  kernels + autograd wrapper)
- `compat/` — separate distribution providing the `flash_attn` alias
- `tests/` — pytest suite (`uv run pytest tests/`)
- `scripts/` — the FA4 race tooling (benches, profiling, the ptxas
  uniform-register probe)
- `CLAUDE.md` — contributor guide; `HANDOFF.md` — the full race log,
  methodology and codegen lessons
