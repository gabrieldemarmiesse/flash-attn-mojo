# Reference PTX — Tri Dao FlashAttention-4

`fa4_fwd_sm90_bf16_hdim128_noncausal.ptx` is the guiding light for the
Mojo kernel: the PTX of FA4's Hopper forward kernel
(`flash_attn.cute.flash_fwd_sm90.FlashAttentionForwardSm90`) for the
canonical config we are trying to match.

The backward counterparts (same canonical config):

- `fa4_bwd_sm90_bf16_hdim128_noncausal.ptx` — the main bwd kernel
  (`FlashAttentionBackwardSm90`). Config from
  `interface._tile_size_bwd_sm90(128, ...)`: tile_m=80, tile_n=128,
  SdP_swapAB + dQ_swapAB, 2-stage Q/dO/PdS pipelines, dQ drained via
  a smem mailbox + `cp.reduce.async.bulk...add.f32` (20480 B per
  warpgroup per m-tile, issued by a producer warp; there are NO
  `red.*` atomics anywhere in the PTX). PTX = one unrolled
  iteration: 24x `wgmma.m64n80k16` (S^T, dP^T, dQ^T) +
  10x `wgmma.m64n128k16` (dV, dK; k=80 -> 5 each).
- `fa4_bwd_preprocess_bf16_hdim128.ptx` — dpsum = rowsum(dO*O),
  lse_log2 = lse * log2(e), zero dq_accum.
- `fa4_bwd_postprocess_bf16_hdim128.ptx` — dq_accum fp32 -> dq bf16.

Backward target numbers (H100 PCIe, B=2 S=8192 H=16 D=128,
non-causal, torch.profiler CUPTI):

| kernel | time |
|---|---|
| FlashAttentionBackwardSm90 (main) | 6198 us |
| preprocess | 153 us |
| postprocess | 109 us |
| **bwd total** | **~6460 us (~425 TFLOPS over 2.75 TFLOP)** |

## Causal counterparts (2026-06-10)

- `fa4_fwd_sm90_bf16_hdim128_causal.ptx` — same FwdConfig(128, 128,
  RS, overlap) as non-causal; 48 wgmma sites (separate masked loop —
  0 trips at sq==sk — plus a steady loop that still replays the
  232-instr mask block every tile); `SingleTileLPTScheduler` (flat
  1-D grid, heaviest-m-first, L2 swizzle-8 head grouping).
  Target: ~1238 µs at the canonical shape (locked 1500 MHz clocks).
- `fa4_bwd_sm90_bf16_hdim128_causal.ptx` — tile_m=64, tile_n=128,
  dQ_swapAB=False (dQ = dS·K, per-wg 64-column split, trans(0,1));
  24x m64n64k16 + 8x m64n128k16 per iteration; plain 3-D grid (no
  LPT for bwd). Target: ~3129–3204 µs main kernel.

## Canonical config ("the simplest call")

```python
from flash_attn.cute import flash_attn_func
out, lse = flash_attn_func(q, k, v)   # all defaults: non-causal, scale=1/sqrt(D)
```

- dtype **bf16**, head_dim **128**, **non-causal**, fixed seqlen,
  no MQA/GQA (Hq == Hk), no window/softcap/alibi/dropout/paged-KV.
- Canonical benchmark shape: **B=2, S=8192, H=16, D=128**
  (`(batch, seqlen, nheads, head_dim)`, contiguous).
- FLOPs: `4·B·H·S²·D = 1.10 TFLOP` per fwd.

## Target numbers (H100 PCIe, this box, 2026-06-10)

| metric | value |
|---|---|
| kernel time (torch.profiler CUDA total / iters) | **2186 μs** |
| achieved throughput | **~503 TFLOPS bf16** |
| kernel | single launch of `FlashAttentionForwardSm90`, sm_90a |

## What the FA4 sm90 kernel does at this config

From `flash-attention/flash_attn/cute/interface.py::_tile_size_fwd_sm90`
and `flash_fwd_sm90.py`:

- tile_m=128, tile_n=128, head_dim tile = 128.
- `mma_pv_is_rs=True` — the P·V WGMMA takes A (=P) from registers.
- `intra_wg_overlap=True` — softmax of tile *i* overlaps the GEMMs of
  tile *i±1* within a warpgroup.
- Warp-specialized: 1 producer warpgroup (TMA loads of Q/K/V) +
  2 MMA warpgroups → 384 threads/block; `num_stages` smem pipeline
  for K and V, mbarrier-synchronized.
- PTX op mix to match: 32× `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`,
  TMA `cp.async.bulk.tensor.4d` for loads (10) and stores (2),
  `mbarrier.*` sync, `ex2.approx.ftz.f32` softmax.

## Regenerating

```bash
scripts/master_bench.sh --refresh-fa4-ptx
# or by hand:
CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=reference_ptx \
FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$(mktemp -d) \
  .venv/bin/python -c "...one fa4 fwd call..."
# then: tr -d '\000' < cutlass*.ptx > fa4_fwd_sm90_bf16_hdim128_noncausal.ptx
```

The PTX is shape-independent (B/S/H are dynamic; only dtype, head_dim,
causal-ness and tile config are baked in), so one dump covers every
shape of the canonical config. Versions that produced it:
`flash-attn-4 == 4.0.0b16`, `nvidia-cutlass-dsl == 4.5.2`, CUDA 12.9
nvvm (the DSL's bundled toolchain), driver/GPU: H100 PCIe sm_90a.
