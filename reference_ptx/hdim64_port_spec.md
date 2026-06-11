# FA4 hdim64 Port Spec — Synthesis & Cross-Check

## 0. Cross-check verdicts (PTX wins)

All four reports were diffed against the committed PTX (`/root/flash-attn-mojo/reference_ptx/`). Nine contradictions found and resolved:

- **[C1] fwd PV k-steps.** Reader C claims "4 calls per PV compute (64/16=4 steps)"; Reader A claims 8. **PTX: Reader A is right.** `fa4_fwd_sm90_bf16_hdim64_noncausal.ptx` contains exactly 8× `m64n128k16` (QK) + 16× `m64n64k16` (PV); per steady kv-tile the loop body issues 4 QK + 8 PV (PV contracts over **k=BN=128 → 8 k16 steps**; n=64 is the *output* width, not the contraction). FLOP check: per WG, QK = 64·128·64·2 = PV = 64·64·128·2. Reader C confused k with hdimv.
- **[C2] fwd hdim128 K/V stage count.** Reader A claims 3 stages (~148–165 KB); Reader C claims `num_stages=2` (hardcoded, `interface.py:857`). **PTX: Reader C is right.** `fa4_fwd_sm90_bf16_hdim128_noncausal.ptx` has 10 `mbarrier.init` (the 11th grep hit is `fence.mbarrier_init`), and smem bases V@1024 (2×32768), Q@66560, K@99328 (2×32768) → total 164,864 B, 2 stages each. Note: **our** Mojo hdim128 fwd uses a 6-slot (3K+3V) ring — that's our own deeper design, not FA4's.
- **[C3] bwd dQ wgmma form.** Reader B says dQ is RS at lines 2044–2111 with B=V; Reader D says SS. **PTX: dQ is SS**, and it's the *middle* n64 group (lines 1921–2007): A descriptor base = `%r412 + 101376` (**sdS**, stage offset `%r905<<11` desc-units = ×32 KB), B descriptor base = `%r400 + 52224` (**K**), `p,1,1,0,1` (transA=0, transB=1). The RS group at 2042–2111 is **dK** (B base `+3072` = Q stage, line 2021); the RS group at 1822–1890 is **dV** (B = dO). Reader B swapped dQ↔dK line attributions and got dQ's operands wrong; Reader D had the right form/operands but mislabeled the same lines "dK SS path" in his §2.
- **[C4] bwd S-GEMM operands.** Reader B says A=Q, B=K. **PTX: A=K, B=Q** — the S wgmma at line ~826 takes `%rd33` (A, base `%r259+52224` = K, line 809) and `%rd34` (B, base `%r248+3072` = Q, line 792). Consistent with SdP_swapAB: each WG owns m64 = 64 **KV** rows × n128 = all 128 Q columns. Reader D right.
- **[C5] bwd "2-CTA cluster".** Reader B claims a 2-CTA cluster for cross-CTA lse/dpsum sharing. **Wrong.** No `.reqnctapercluster`/`.explicitcluster` directive exists; every `mapa.shared::cluster` maps to the CTA's **own** `%cluster_ctarank` (identity, e.g. lines 211–216); the identical 8 `mapa` pattern exists in the hdim128 bwd PTX, which we already matched at parity without clusters. It's CuTe-DSL boilerplate for 1-D `cp.async.bulk` cluster-space addressing.
- **[C6] bwd smem swizzle.** Reader D claims SWIZZLE_64B (`(addr>>3)&56`) at hdim64. **Wrong.** Every wgmma descriptor or-constant in both hdim64 kernels (e.g. `4611686293305360384` = 0x4000004000010000) has layout_type bits [63:62] = 1 = **SWIZZLE_128B**; the and-mask histograms show `&112` (fwd64 ×4, bwd64 ×3) and **zero** `&56`. A 64-wide bf16 row = 128 B = exactly one SW128 period (Reader C's bit calc was correct). **All our 128B-swizzle XOR math, TMA `swizzle_mode`, and stmatrix step rules carry over unchanged.**
- **[C7] dV/dK RS A-fragment provenance.** Reader B says they're "loaded from the transposed sdS smem (reads back stmatrix.trans output)". **Wrong — `grep -c ldmatrix` = 0** in the entire bwd kernel. The RS A operands come straight from accumulator-derived bf16 registers; the *same* registers `%r316–347` are both stmatrix'd to sdS (lines 1798–1805, for dQ's SS read) and used directly as dK's RS A operand (line 2044+).
- **[C8] fwd producer's role in scheduler barriers.** Reader A says "the producer walks bars 2→3→4, signaling each consumer after filling its K/V slot". **Wrong** — the WarpScheduler barriers are a consumer-only ring (count 256 = 2×128 consumer threads: owner syncs, predecessor arrives; `flash_fwd_sm90.py:1524–1545`, PTX `selp %r70, 2, %r25+2, %r25==3`). Producer↔consumer flow control is the K/V mbarrier ring only. Reader C right.
- **[C9] minor.** Reader B's "4 lse/dpsum slots = 2 stages × 2 wgs" — actually 2 sLSE stages (offsets 128, 640) + 2 sdPsum stages (1152, 1664), matching stages_Q/dO=2 (Reader D). Reader A's "10 mbarriers" is correct (grep `mbarrier.init` count 11 includes the fence line).

Everything else cross-checks clean: fwd hdim64 = 512 thr, setmaxnreg 32/160, BM=192/BN=128, K/V 2+2 stages of 16,384 B, Q/O 24,576 B @33792 overlay, total 91,136 B, 10 mbarriers, 2 LSE `st.global.f32`/thread (ln-domain, ×0f3F317218); bwd hdim64 = 384 thr, 24/240, BM=BN=128 both causal and noncausal, smem 199,680 B with the offset map as tabulated by B/D (verified at 3072/35840/52224/68608/101376/166912/183296), 4+4 SS m64n128k16 (4 k-steps, k=D=64) + 24 m64n64k16 (8 k-steps each), 2× `cp.reduce...add.f32` of 16,384 B at smem 166912/183296, forward m-loop, plain 3-D grid `(ceil(Sk/128), H, B)`, causal = leading-tile skip `max(floor((n*128+Sq−Sk)/128),0)` + 1 predicated diagonal tile. `shuffle_LSE = SdP_swapAB and tile_hdim<=64` → True (`flash_bwd_sm90.py:123–124`; 66 `shfl.sync.idx` in bwd64 vs 2 in bwd128).

---

## 1. DELTA TABLE

Conventions: "ours today" = hdim128 kernels in `src/flash_attn_mojo/{fwd_fa4,bwd_fa4}/`. All new behavior is comptime-forked on `head_dim`; **hdim128 PTX must stay byte-identical at every step** (dense byte-gates).

### Forward

| Aspect | FA4 hdim64 | Our hdim128 today | Port action |
|---|---|---|---|
| Threads / WGs | 512 = 1 producer + **3** MMA WGs (`.reqntid 512`) | 384 = 1 + 2 (`common.mojo: kFa4NMmaWarpgroups=2`) | `kFa4NMmaWarpgroups` → comptime fn of head_dim: 3 at 64. Grid stays `(ceil(S/BM), H, B)` |
| setmaxnreg | dec **32** / inc **160** (PTX lines 116/346; `interface` table `{3:(160,32)}`) | dec 24 / inc 240 (`warpgroup_reg_dealloc/alloc`, kernel.mojo:271/341) | comptime NUM_PRODUCER/CONSUMER_REGS by head_dim. Pool: 3·128·160+128·32 = **65,536 = exactly the 64K regfile** |
| Tiles | BM=**192**, BN=128 (`FwdConfig(192,128,RS,OL)`, unconditional causal+noncausal, `interface.py:129–134`) | BM=128, BN=128 | `kFa4BlockM` comptime by head_dim |
| K/V ring | 2+2 stages × 16,384 B; total smem 91,136 B | 6 slots (3K+3V) × 32,768 B; ~224 KiB | Keep our 6-slot ring at 16,384 B/slot (proven pipeline; total 24,576 + 6·16,384 + 128 = **123,008 B**, fits). Optionally A/B vs 2+2 later |
| Q/O smem | 24,576 B (192×64×2), O overlays Q @33792 | 32,768 B, O overlays Q | Sizes follow comptime; overlay still exact (both 24,576 B) |
| Swizzle | **K_SW128 unchanged** (PTX `&112`, descriptors layout_type=1) [C6] | SW128 | **No change** — 64-wide bf16 row = one 128-B period; all XOR/stmatrix/TMA swizzle code carries |
| QK wgmma | m64n128k16 SS × **4** k-steps | m64n128k16 SS × 8 | comptime k-step count = head_dim/16. S-acc stays 64 f32/thread, 2 rows/thread |
| PV wgmma | m64n**64**k16 RS × **8** k-steps (k=BN) [C1] | m64n128k16 RS × 8 | n=64 arm: bf16 stdlib has it (`modular/.../mma.mojo:1083 elif n==64`); **f16 vendored emitter `_wgmma_f16.mojo` covers only m64n128k16 — add a `wgmma_rs_f16_m64n64` arm** (c=SIMD[f32,32], 32-reg constraint list). O-acc halves: 32 f32/thread. P c→a frag cast stays valid (num_m_mmas=1) |
| Scheduler pingpong | **3-way ring** WG2→WG0→WG1→WG2; sync id = wg+2 (=2,3,4), arrive id = `selp(2, wg+3, wg==3)`; count 256; only WG1 pre-arrives at init [C8] | 2-way: sync `1+wg`, arrive `2-wg`, ids 1/2 (kernel.mojo:354/578/591) | Generalize arrive to `(wg+1)%NWG + 1` via comptime/selp (NOT runtime `Int %` — signed-div chain). At NWG=2 the formula must reduce to today's PTX byte-identically |
| Epilogue barrier | id 1, count 384 (stmatrix) then 416 (=384+32, producer warp joins TMA fence); store by warp 4 | id 3, count NWG·128 (kernel.mojo:744) | hdim64 fork: sched ids 1,2,3 + epilogue id **4**; counts NWG·128 / NWG·128+32 |
| Softmax / LSE | 2 rows/thread, 32 ex2/row, 2 `st.global.f32` (+32 B apart), natural-log | identical | **No change** except a `row < seqlen` predicate on LSE stores in tail m-tiles (new — see TMA row) |
| Q/O TMA & tails | True 4-D (S its own dim): OOB loads zero-fill, OOB **stores clamp**; 8192 % 192 = 128 → canonical shape has a 128-row tail tile | 3-D flattened (B·S) — interior-batch tails read next batch (benign at BM=128: no tails exist) | **Mandatory: give S its own TMA dim for Q and O at hdim64.** With flattening, the tail O-store would *write* the next batch's rows 0–63 — a store-side clobber the +inf-LSE trick cannot fix. K/V (BN=128, S%128==0, exact) can stay as-is |
| Causal | Same 192×128 config; LPT decode (7 extra params); masked band spans **2** kv-tiles (BM=192 > BN=128); 3 nounroll loops | LPT; 1 diagonal tile | Reuse LPT host machinery with BM=192; generalize the masked phase from "single diagonal tile" to a ≤2-tile masked loop |
| mbarrier counts | consumer-done count 12 = NWG·4 | 8 = NWG·4 | comptime formula, no structural change |

### Backward

| Aspect | FA4 hdim64 | Our hdim128 today | Port action |
|---|---|---|---|
| Threads / roles / regs | 384, 1 prod + 2 MMA, 24/240 — **identical** | identical | No change (pool stays 384·168) |
| Tiles | BM=**128** noncausal AND causal, BN=128 (`interface.py:177–184`) | BM=80 noncausal / 64 causal | One BM for both — the causal/noncausal BM fork disappears at hdim64. Dense becomes **exact-fit** (S%128==0 → zero partial m-tiles, no pad rows) |
| Stages | Q 2, dO 2, PdS 2, K/V 1 — same counts | same (kBwdQdOStages=4 interleaved) | Slot sizes only: Q/dO 16,384 B, lse/dpsum 512 B, sdS 32,768 B/stage |
| smem total | **199,680 B** (map verified: sLSE@128, sdPsum@1152, sQ@3072, sV@35840, sK@52224, sdO@68608, sdS@101376, sdQaccum@166912+183296) | 227 KB (exact cap) | 27.8 KB headroom appears; do NOT spend it in v1 (FA4 ships 2-stage) |
| S^T/dP^T | m64n128k16 SS × **4** k-steps (k=D); A=K/V, B=Q/dO [C4]; 64 c-regs | m64n80k16 SS × 8; 40 c-regs | n: 80→128 (drops the StaticTuple n=80 path — plain SIMD[f32,64]), k-steps 8→4 |
| dV / dK | m64n**64**k16 **RS** × 8 k-steps (k=BM=128); A = P^T/dS^T regs direct (no ldmatrix [C7]); 32 c-regs | m64n128k16 RS × 5 (k=BM=80); 64 c-regs | comptime n & k-step count; **f16 needs the new m64n64 emitter arm** (same as fwd PV) |
| dQ path | **Unswapped SS** dS@K: A=sdS, B=K transB=1 [C3]; AtomLayoutMdQ=2 = per-WG 64-**BM-row** A-window (+16,384 B, swizzle-atom clean); 8× m64n64k16, 32 c-regs; sdQaccum (4096,2) f32, 2× cp.reduce 16,384 B | **Swapped** dQ^T = K^T·dS^T, m64n80k16 SS (causal: m64n64k16); per-WG **D-slab (M) split**; single 40 KB mailbox; 1 drain | **KEEP the swapped dQ^T** — see verdict below |
| sdS staging | 2× 32,768 B; stmatrix.trans ×8/WG, +2048 B steps (safe per our step rule); SW128 | 2× 20,480 B (128×80) | Resize; orient the staged tile so the per-WG dQ B-window is a **row-block** offset (+16,384 B = 16 swizzle atoms) |
| LSE/dPsum regs | shuffle_LSE/dPsum = True (66 shfl.sync.idx) — frees ~14 regs at 240 budget | read lse_log2/dpsum from smem ring at use sites | Keep our smem-read approach first; shuffle_LSE is a fallback lever only if the spill canary fires |
| Epilogue (MHA) | dK/dV bf16 staged in dead K/V smem (16,384 B each, exact), 1 TMA S2G each | same pattern at 32 KB each | Scales directly |
| Epilogue (GQA) | n/a (FA4 ref is MHA) | dK/dV staged as f32 in dead K+V smem = exactly 64 KiB | **Breaks at hdim64**: dead K+V = only 32 KB but dK,dV f32 = 2×32 KB. Stage in the dead **sdS** (2×32 KB, exact fit, dead after the last dQ GEMM + wait) |
| Preprocess / convert | dq_accum `(B,H,ceil(S/128)·128,64)` f32; zero 8192 f32/m-tile; lse·log2e/dpsum padded to %128 | `(B,H,ceil(S/80or64)·BM,128)`; kBwdPreBlockM=128 already; convert 256 thr × 64-elem half-row | Pre: D=64 row reductions, zero 32 KB/m-tile; dense pad = none (Spad=S). Convert: new m64n64 transposed-fragment decode, one 64-elem row per thread |
| Host allocs | — | `bwd_fa4/__init__.py:88–99`: `block_m = 64 if causal else 80`, `seqlen_pad=ceil(S/block_m)·block_m`, dq_accum B·H·seqlen_pad·128 | `block_m=128` both; **no 80-padding anywhere**; dq_accum = B·H·S·64 f32; dpsum/lse_log2 = (B,H,S); GQA dk/dv_accum (B,Hkv,S,64) f32 |
| Scheduler | plain 3-D grid, forward m-loop, no LPT | same | No change |
| Varlen tables | padded_offset_q with BM=128 | builders parameterized: `bwd_fa4/__init__.py:172–245` take `block_m`; fwd `_BLOCK_M=128` (`fwd_fa4/__init__.py:102`) | bwd: pass 128 (both causal/noncausal) — code already generic. fwd: BM=192 breaks the exact-fit envelope (seqlens %128 but not %192 → in-pack tails that TMA cannot clamp). **v1: run hdim64 varlen fwd at the (128,128, 2-WG) config — FA4's own blessed fallback (`interface.py:131–133`)**; tail-predicated 192 varlen is the follow-up |

### dQ verdict: KEEP the swapped dQ^T (with the split moved M→N)

At hdim64, dQ^T = K^T·dS^T has output (D=64 × BM=128): D is now a **single m64 atom**, so our hdim128 per-WG D-slab (M) split is impossible. Move the split to N: each WG computes `dQ^T[:, wg·64 : wg·64+64]` as **m64n64k16 SS × 8 (k=BN=128)** — A = K^T (full, shared by both WGs; our existing hand-rolled `wgmma_async[layout_a="col"]` descriptor, kernel.mojo:202), B = a per-WG 64-BM-column window of staged dS^T. This is the exact mirror of FA4's AtomLayoutMdQ=2 BM-row split: **identical instruction (m64n64k16 ×8 — already what our causal hdim128 dQ^T emits), identical c-regs (32 f32), identical mailbox traffic (2×16 KB = 32 KB/m-tile vs FA4's (4096,2)+2 cp.reduce), identical sdS round-trip** (FA4's unswapped form needs sdS for its A operand just the same). The causal-hdim128 algebra argument carries verbatim. One new constraint: the per-WG B-window must land on a swizzle-atom boundary — stage sdS with BM as the outer (row) dimension so the window is `+64 rows × 256 B = +16,384 B` (clean), not a 128-B intra-row column offset (which would shift the SW128 XOR phase). Mailbox: one contiguous 128×64 f32 = 32 KB buffer, two disjoint 16 KB WG halves, drain warp does one 32 KB `cp.reduce` per m-tile — protocol byte-for-byte our existing one. Convert kernel decodes the transposed fragment dump as today, re-parameterized to (D=64, BM=128).

---

## 2. ORDERED PORT CHECKLIST

Protocol wrapper for **every** step: `sudo nvidia-smi --lock-gpu-clocks=1500,1500` at session start; after every kernel edit run `ptxas -arch=sm_90a -v ptx/mojo_*.ptx` (spill bytes must be 0 on **all** variants); bench only interleaved via `master_bench.sh`, quote 3-run spreads. Captured FA4 targets at **B=2, S=8192, H=32, D=64**: fwd **2798** µs noncausal / **1596** causal; bwd **7642** / **3853**.

**Step 0 — Scaffolding + byte-gate harness.**
Thread `head_dim` as a `-D` define through `_jit.py`/`variant.mojo` for both subpackages; lift `_SUPPORTED_HEAD_DIM` (`_fn.py:30`) to `{64,128}`; add `--dims`-style shape plumbing to `bench_fa4.py` (default stays 2,8192,16,128) and a `--hdim 64` mode to `master_bench.sh` mapping to the committed `reference_ptx/fa4_*_hdim64_*.ptx` for the op-mix diff. Dump baseline PTX for the FULL existing hdim128 matrix (fwd: plain/causal/gqa/varlen/fp16; bwd: same + preprocess/convert).
**GATE:** all hdim128 PTX byte-identical to baseline; `uv run pytest tests/`; `--check-only` fwd+bwd green; one interleaved hdim128 parity bench unchanged.

**Step 1 — Comptime parameterization sweep (no hdim64 yet).**
Convert `kFa4BlockM`, `kFa4NMmaWarpgroups`, reg counts, smem formulas, mbarrier/barrier participant counts, wgmma k-step counts, and the scheduler-arrive formula to head_dim-keyed comptime functions, evaluated at 128.
**GATE:** byte-identical hdim128 PTX for every variant (this is the highest-risk refactor for silent codegen drift — gate per edit, not per batch).

**Step 2 — Run `scripts/ptxas_ur_probe.py` in the new fwd dataflow shape** (3 consumer WGs, 6-slot ring, 4 QK-step + 4 K-step + 8 PV-step descriptors/iter ≈ 16 live) BEFORE touching the fwd kernel. Confirm UR allocation stays under the cliff with the laundering pattern; record the probe config.
**GATE:** probe shows 0 spills / no R2UR storm at the planned descriptor inventory.

**Step 3 — fwd dense noncausal hdim64.** BM=192/NWG=3/regs 32-160; Q/O TMA descriptors with S as its own dim (tail clamp) + LSE store predicate; 3-way scheduler ring (ids 1,2,3; epilogue → id 4); PV n=64 (bf16 stdlib arm; add `wgmma_rs_f16_m64n64` to `_wgmma_f16.mojo` mirroring the n==128 arm 1:1); O-acc 32 f32.
**GATE:** `--check-only` at S ∈ {128, 256, **384** (exact 2×192), 640, 1024} + LSE check (tails exercised at 128/256/640/1024); 0 spills; hdim128 byte-gate; PTX op-mix diff vs `fa4_fwd_sm90_bf16_hdim64_noncausal.ptx` (expect 4 QK m64n128 + 8 PV m64n64 per iter, `&112` XORs); interleaved bench vs **2798** µs at (2,8192,32,64) — parity = within the 2–4% wobble band over 3 runs.

**Step 4 — fwd causal hdim64.** LPT host params recomputed for BM=192; masked phase generalized to the ≤2-tile diagonal band.
**GATE:** causal check-set incl. tail seqlens; spills 0; byte-gates; bench vs **1596** µs.

**Step 5 — fwd GQA hdim64.** Comptime ratio on K/V TMA coords — mechanics unchanged.
**GATE:** GQA check set (Hq=16/Hkv=4 analog at D=64 → Hq=32/Hkv=8); byte-gates; spot bench.

**Step 6 — bwd dense noncausal hdim64.** BM=128; smem map ≈ FA4's 199,680 B; SdP n=128 ×4 k-steps; dV/dK m64n64 ×8 RS (+f16 arm reuse from step 3); swapped dQ^T with N-split B-windows (+16,384 B row-block offsets) and the 32 KB mailbox; preprocess (zero 32 KB/m-tile, D=64 dpsum) and convert (m64n64 fragment decode) at BM=128; host allocs (no 80-padding; dq_accum B·H·S·64).
**GATE:** `--check-only --kind bwd` S ∈ {128, 256, 640, 1024} (all exact-fit now — keep 640 anyway); 0 spills across main/preprocess/convert; hdim128 byte-gates (incl. preprocess/convert); op-mix diff vs `fa4_bwd_sm90_bf16_hdim64_noncausal.ptx` (8 m64n128 + 24 m64n64, 2 cp.reduce); bench vs **7642** µs.

**Step 7 — bwd causal hdim64.** Same BM=128 config (no causal BM fork); leading-tile skip math at BM=128; single predicated diagonal tile; plain grid (no LPT — PTX-verified).
**GATE:** causal bwd check set; end-to-end autograd test causal D=64; bench vs **3853** µs.

**Step 8 — bwd GQA hdim64.** Move dK/dV f32 epilogue staging from dead K+V (now only 32 KB) to dead sdS (64 KB exact); per-kv-head accumulators (B,Hkv,S,64) f32; preprocess zeroes them; permute-cast convert.
**GATE:** GQA bwd correctness vs fp32 reference + cross-check vs `flash_attn.cute` where importable; byte-gates; interleaved GQA bench.

**Step 9 — varlen hdim64.** bwd: pass `block_m=128` through the existing builders (`bwd_fa4/__init__.py:88/301`), padded_offset with BM=128 — code already parameterized. fwd: compile the hdim64 kernel at the (128,128, NWG=2) fallback config for varlen (FA4-blessed; avoids in-pack tails entirely).
**GATE:** `--check-only --varlen` per-seq fp32 references; canonical mixed 16384-token bench; byte-gates; signatures stay byte-identical to dense (table addresses ride the existing arg slots).

**Step 10 — tests + docs.** Add D=64 cases to `tests/test_fa4.py` and the `--check-only` sets (not ad-hoc scripts); update CLAUDE.md/HANDOFF with the hdim64 parity matrix and the captured targets; note the fwd-varlen 192-tail follow-up alongside task #18.

---

## 3. RISKS

**Codegen hazards**
1. **UR-file at 3 WGs (fwd):** per-iteration descriptor inventory ≈ 16 (4 Q-kstep + 4 K-kstep + 8 V-kstep) sits **at** the documented ≥~16 cliff. The `mov.b32` laundering must stay on every smem root; verify with the probe (step 2) before kernel edits, and SASS-diff for `IMAD.U32 R,RZ,RZ,URx`/R2UR storms after.
2. **tid-widening at 512 threads:** `tid >> 7` now yields 0..3 *including the producer*; consumer wg index = `warp.broadcast(Int32(tid >> 7)) - 1` ∈ {0,1,2}. The ring-arrive `(wg+1) % 3` must be a comptime-unrolled selp (FA4 emits `selp r70, 2, r25+2, r25==3`), never a runtime signed `%`/`//` (17-op floor-div chain). Producer test stays `tid < 128`.
3. **Named-barrier ids:** fwd grows from {1,2 sched; 3 epilogue} to {1,2,3 sched; 4 epilogue} — the epilogue id move must be comptime-forked or hdim128 PTX changes. bwd keeps {4, 6,7,8, 9,10}. Max id 10 ≤ 15; id 0 stays reserved for implicit `barrier()`. No cross-kernel collision possible (separate launches).
4. **Smem arithmetic:** fwd 123,008 B (6-slot) / bwd ≈199,680 B — both clear the 232,448-B opt-in cap. The sharp edge is the **GQA-bwd epilogue exact-fit change**: dead K+V shrinks 64→32 KB while the f32 dK/dV staging stays 64 KB — must move to dead sdS or it silently overflows into live buffers.
5. **fwd TMA tail clobber:** with our flattened (B·S) descriptors, the 128-row tail at S=8192/BM=192 would make interior batches *write* the next batch's O rows 0–63. S must become its own TMA dimension for Q and O (loads zero-fill, stores clamp) — this is correctness, not perf, and the canonical bench shape exercises it.

**Register budgets**
6. fwd pool at 512 threads: 3·128·160 + 128·32 = **65,536 = the entire register file, zero headroom** (hdim128's analog 384·168 = 64,512 left 1,024 slack). Consumer regs cannot exceed 160; budget sanity: S-acc 64 + O-acc 32 + packed P 32 ≈ 128 f32 + stats/addressing — FA4 fits, our single-S/single-P design should too, but the spill canary is the only truth. Don't trust ncu's static registers/thread.
7. bwd stays at 24/240, but per-thread LSE/dPsum coverage grows (n=BM=128 → 32 q-columns per c-fragment vs 20 at n=80). Our smem-read-at-use-site approach scales; if the canary fires, FA4's shuffle_LSE (8-thread shfl.idx distribution, frees ~14 regs) is the documented escape hatch.

**Where hdim64 needs NO new machinery (already-general comptime/host params)**
- **Swizzle — nothing changes** [C6]: SW128 atoms, `addr^((addr>>3)&112)`, the 2048-B-step stmatrix rule (FA4's bwd64 stmatrix steps are exactly +2048), TMA swizzle modes.
- GQA comptime ratio (fwd K/V coords; bwd per-kv-head reduction protocol).
- LPT scheduler machinery (host-side param recompute only; bwd confirmed LPT-free at hdim64 too).
- The dQ mailbox/drain-warp/cp.reduce protocol and the convert-kernel pattern (re-parameterized, not redesigned).
- mbarrier ring conventions, lse/dpsum riding stage mbarriers, the varlen arg-slot riding trick and table-builder code paths (bwd builders already take `block_m`).
- Cache invalidation (`_jit_common.py` env signature), `MOJO_DUMP_PTX` plumbing, and the whole measurement/profiling toolchain — only shape flags and reference-PTX mappings need extending.