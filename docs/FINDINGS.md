# Findings: GLM-5.3-Flash EXL3 + DFlash2 on 2× GB10

Scope: everything here was measured on two DGX Sparks (GB10, 128 GiB unified
memory each) over a direct 200GbE RoCE link, serving
brandonmusic's EXL3/TR3 4bpw quant at 131,072-token context, temperature 0
unless stated. Numbers are 3-run medians where the harness supports it.

> **Credit**: the stack builds on the local-inference-lab vLLM fork lineage
> (b12x runtime, breakable CUDA graphs, kpool sparse attention), eugr's
> build system, and tonyd2wild/jack6464's forum groundwork for
> GLM-5.3-Flash-on-Spark generally. What this project adds: the glm5_next
> port onto that fork, EXL3 fused serving from the standard checkpoint, the
> DFlash2 transplant + ring draft-KV, the quality gates, and the memory
> methodology. Headline delta over the best prior config on the same kit
> (NVFP4 + MTP-4): +21% single-stream, +6pt gpqa, vision fixed.

## 1. Getting EXL3 to serve at all

The checkpoint is "generic" EXL3 (per-expert tensor_storage), which the
fork could load but only through a per-expert GEMM loop: 2.8 tok/s. Two
steps fixed that:

- **Fused trellis path from the standard checkpoint.** The fork's fast path
  wanted rank-segmented tensors. A load-time TP-slicing port
  (`standard_fused_moe`) builds the b12x fused runtime directly from the
  standard checkpoint — no offline conversion. 2.8 → 27.6 tok/s.
- **CUDA graphs on top**: 27.6 → 28.6 (MTP-4), with the fork's breakable-
  graph machinery handling the speculative loop natively. **Fix** commits
  are indexed in the README's under-the-hood table.

What did *not* work first: serving the quant under the NVFP4 recipe's
`--moe-backend marlin` (wrong backend class for EXL3 — crashes the worker),
and MTP on mismatched quants (the two NVFP4 community quants package MTP
experts differently; they are not interchangeable).

## 2. DFlash2: the transplant and the head-to-head

The fork predates upstream's DFlash2 lane, so the lane (draft model,
speculator, selector walk) was transplanted from upstream vLLM onto the
fork's V1 runner — 14 files plus three glm5-specific pieces that do not
exist anywhere upstream:

1. **mHC aux capture.** GLM-5.3's hyper-connection residual (n-stream,
   deferred `hc_post`) is incompatible with the stock EAGLE3 capture
   contract. Capture materializes `hc_post` then `hc_contract` at the layer
   boundary — validated against a numerical oracle for all five tap layers
   before any serving work.
2. **Draft KV grouping.** The drafter's plain sliding-window specs knocked
   the model off the glm5 KV fast path onto a generic path that provably
   cannot serve this hybrid (kpool page unification breaks). The drafter's
   layers now get their own groups inside the glm5 lane.
3. **Non-causal in-block draft attention.** The drafter's checkpoint says
   `is_causal: false`; a causal mask inside the draft block silently
   collapses later-position acceptance. (The parallel MiaAI recipe hit the
   identical bug via a different backend — see COMPARISON.md.)

Result, single variable against the native MTP head on identical serving:

| | MTP-4 | DFlash2 k=7 |
|---|---:|---:|
| c1 prose | 28.6 | **33.1–34.7** (+21%) |
| c5 aggregate | 91.7 | 83.7–96.0 |
| TTFT | 0.42 s | 0.50 s |
| accept | 3.80/5 | 3.94/8 (prose) |
| math_500 | 90% | 90–94% |

## 3. Ring draft KV: recovering the pool

DFlash2's five sliding-window (2048) draft layers initially allocated KV
across the *whole* pool despite each request touching at most a couple of
blocks — 710k tokens (MTP-4) fell to 329k. The fix: draft tensors are a
fixed ring of `1 + max_seqs × R` pages, with a Triton kernel positionally
remapping draft block tables each step (`phys = 1 + req×R + pos mod R`).
Draft layers opt out of prefix caching; restored-prefix shifts compose with
the remap because position-encoded ids move with positions. Pool: 329k →
**520,470** tokens at the same budget; acceptance unchanged through ~48
ring wraps at 112k depth. Kill switch: `VLLM_DFLASH_KV_RING=0`.

(The MiaAI recipe solved the same problem by slot-sharing the drafter into
its MLA tensors — an exact-fit rescale their KV geometry admits. Two
independent designs, same ~zero-cost outcome.)

## 4. Equivalence methodology — read this before "verifying" speculation

Strict token-equality between spec-on and spec-off runs **fails on healthy
stacks**: batch-shape numerics flip argmax ties (≤0.75 nats), and the
speculative server is run-to-run non-deterministic (accept-trajectory-
dependent kernel shapes) while the baseline is deterministic. Worse,
single-position logprob probes with `continue_final_message` fabricate
multi-nat "smoking guns" at token-boundary re-tokenizations.

The protocol that works ([tools/dflash_equiv.py](../tools/dflash_equiv.py)):
rescore *full* outputs under the same server with `prompt_logprobs`, flag
positions with logprob < −2 or rank > 3, and compare the spec run's flag
rate against the baseline's own control. Verdict on this stack: 47/3068 vs
40/3069 flags — **lossless up to argmax tie-flips**.

## 5. Memory: floors, budgets, and the head-swap result

GB10 unified memory swap-wedges rather than OOMs, so every budget claim
here is backed by a 1 Hz `MemAvailable` sampler on both boxes under
saturation + 112k prefill ([tools/memlog.sh](../tools/memlog.sh)):

- Pinned KV budget 12.4 GB → **5.25 GiB measured floor** on the binding box.
- 13.4 GB → floor collapsed to 2.26 GiB for +47k tokens. Rejected. Explicit
  `kv_cache_memory` bypasses vLLM's profiling reserve entirely (the code
  says so) — budget increases are non-linear in floor cost.
- Floors age ~1.5–2 GiB per day of workload. Fresh-boot floors flatter.
- **Head/worker swap is impossible on the reference kit** (3 attempts): the
  box holding weights on local NVMe dies at ~92% of shard load when it also
  runs the head — local read outruns page-cache reclaim; the NFS-paced head
  survives. This is why `--nfs` exists and why the default-local mode
  carries a wedge warning. torch's 600 s TCP rendezvous timeout also bounds
  any boot-staggering workaround.
- The drop-caches ritual before launch is mandatory on both boxes.

## 6. fp8 KV: gated in as an option, not the default

`KV_DTYPE=fp8_e4m3` (+ the FlashInfer sm12x patches baked in the image)
gives 769,817 tokens (+48%) at the same budget. Quality gate at n=100:
**89%** math_500 against a pre-agreed ≥88% bar — so fp8 is a supported
*option* for long-context/many-stream use. bf16 stays the default: an
earlier MTP-4-era fp8 run measured 86% at n=50, and the default config
should not carry that ambiguity. Accept and throughput are noise-identical
between the two.

## 7. Acceptance characterization

Acceptance is workload-dominated and nothing else moves it much:

- Prose ~3.5/8 (the drafter's floor) → JSON 4.7 → math 5.8 → long-context
  code 5.2 → doc-grounded/multi-turn 5.3–7.0 → pure counting 7.8.
- Temperature and thinking mode are **not** levers (2×2 A/B: means
  4.58–4.80, all within noise). Math at temp 1 accepts 6.8–7.1/8, which
  also end-to-end validates the salted-Gumbel sampled path.
- Vision requests show **no acceptance penalty**: the drafter runs
  text-only inputs and inherits visual context through the target's aux
  hiddens — vision is effectively grafted for free.
- Prefix-cache restores do not hurt drafting (restore granularity is whole
  2304-token blocks, divisible by every kernel block size in play).

## 8. Things that did NOT help

| Tried | Result | Why |
|---|---|---|
| MTP-5 | c1 −10% vs MTP-4 | position-5 prose acceptance ~0.23 |
| `EXL3_PREFILL_CHUNK` sweep | ≤3% | trellis GEMM is not the prefill bottleneck |
| Temp/thinking as acceptance levers | null | measured 2×2, all within noise |
| Head/worker swap | 3× boot death | local-NVMe read vs unified-memory reclaim (§5) |
| KV 13.4 GB | rejected | floor collapse (§5) |
| Strict token-equality equivalence | misleading | tie-flips + nondeterminism (§4) |
| MXFP8 DFlash2 draft ([local-inference-lab checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8)) | boot crash at draft load | this fork's draft-side quant-config hydration is exl3-only, so the draft builds unquantized and the checkpoint's `weight_scale` params have no home (first at `candidate_selector.hidden_projection`). Newer fork lineages load it; porting the hydration + quant-aware DFlash2 module construction is parked |

Kept anyway although null: `EXL3_PREFILL_CHUNK` passthrough (free knob).

## 9. The thinking-off template bug

The bundled multimodal chat template unconditionally opened `<think>` in
the generation prompt, so `enable_thinking: false` was silently ignored:
the model always reasoned and the parser misfiled that reasoning into
`content` (with the closing marker stripped — answers like "391391").
Fixed by gating the generation prompt (`<think></think>` closed + no
reasoning-effort hint when thinking is off; handling ported from the MiaAI
recipe's template). Side effect worth knowing: thinking-off workloads no
longer carry a reasoning preamble, which *raised* structured acceptance
0.92 → 0.98 (+7% tok/s). With thinking on, reasoning arrives in
`message.reasoning` — note the field name (not `reasoning_content`).

## 10. The 500k default (2026-08-29)

The default moved from 131k/bf16 to **524,288-token context with
`KV_DTYPE=fp8_e4m3`** — 1,435,070-token pool (2.74 concurrent full banks)
at the *unchanged* 12.4 GB budget. Three findings made it cheap:

- **The block-size dividend.** At fp8, vLLM auto-bumps the KV block
  2304→4608 to keep the KDA linear-state page equal to the attention page
  ("Setting attention block size to 4608" in the boot log). Effective
  per-token cost fell 23.8 KB → 9.7 KB — nearly half of it was the
  fp32 KDA state page, whose per-token share halves with the bigger block.
  The old 131k-era fp8 figure (769,817 tokens) predates this rebalance.
- **MNBT 8192 survives 133k prefill** on this lane (~1,400 tok/s). The
  MiaAI recipe's 1024-chunk cap is a FlashInfer-SM120-lane constraint that
  does not transfer here.
- **`SKIP_MM_PROFILING=1`** is required at long MAX_LEN — the max-size
  multimodal dummy profile OOMs GB10 unified memory.

Quality, measured on the shipped config: **long context is at parity**
(estonia 133,186-token retrieval 9/10 twice — community band 29–30/30;
lavd ledger audit n=30 EXACT 5/NEAR 23/FAIL 2 vs bf16's 6/21/3; 111k
needle all-facts-exact), acceptance 4.51/7 and structured decode
74.2 tok/s (≥ the bf16 72.4). **math_500: parity, settled by a
same-day, same-harness n=100 A/B** — fp8@524k **87/100** (88.0% pooled
with a 44/50 run), bf16@131k **86/100**. fp8 KV costs nothing on math.
The oft-quoted 94% (n=50) belongs to an earlier measurement era (template
distribution + n=50 noise) and should not be used as a baseline; the
current-era band for this stack is **86–88%** at either KV dtype,
thinking off, temp 0. bf16 cannot serve the long banks anyway (0.99× one
524k request).

Long-context validation used
[local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
(`--test-profile estonia` / `lavd`, `--reasoning-effort high`).

## 11. The 1M-declaration wall (negative result, 2026-08-29)

A native-1M `--max-model-len` boot fails on this lane at warmup:
`persistent_topk would oversubscribe ... FilteredTopK fallback requires
>=128KB smem per block (have 101376). total_ctas=90 > num_sms*occupancy=48
(TopK=512)`. The sparse-indexer decode top-k sizes its persistent grid
from the declared max context; GB10's 48 SMs and ~99 KB/SM shared memory
cap the launchable declaration at just above the 524,288 default (~45
CTAs). Not an MNBT/prefill issue and not memory-pool-related — the pool
at a 1M declaration actually grew to 1,658,303 tokens before the kernel
wall hit. Escapes: a sm121-aware multi-pass top-k (upstream-PR candidate)
or the b12x selector lane. Until then 524,288 is the validated ceiling,
and it is a *kernel* ceiling, not a capacity one.

## 12. The GLM_NEXT b12x lane (2026-08-30)

Upstream b12x defines `ModelType.GLM_NEXT`: GLM-5.3-Flash's absorbed
NoPE MLA as a first-class contract — full 512-wide query, packed
**528 B/token** cache record (512 E4M3 latent + 4 inline fp32 group-128
scales, no RoPE payload), FP8 compute, explicit model identity on every
kernel call (512 is shape-ambiguous with DSV4). We back-ported the two
upstream kernel commits onto the shipped b12x
([Entrpi/sparkinfer-glmrt](https://github.com/Entrpi/sparkinfer-glmrt)
branch `glm-next-backport`; two one-line fork fixes: smem_mg GLM routing
and the prefill_mg contiguous early-return; 26/26 GLM_NEXT + 234/234
serving-lane kernel tests on GB10) and grafted the lane into this fork's
`B12X_MLA_SPARSE` backend (fork @ `5c9e2bfd2`): the kpool indexer stays
byte-identical to the fp8_e4m3 lane, its logical top-k indices are
converted to physical slots exactly as the FlashInfer lane does, and the
selector buffer is 2112 wide (the GLM_NEXT prefill contract enumerates
widths — 128-rounding to 2176 is rejected).

Serve it with `ATTN_BACKEND=B12X_MLA_SPARSE KV_DTYPE=fp8_ds_mla
KV_SKIP_LAYERS=sliding_window`. Full gauntlet vs the fp8_e4m3 lane at
identical settings: math_500 **91/100** (vs 87), lavd n=30
**EXACT 15/NEAR 14/FAIL 1** (vs 5/23/2), estonia **10/10** (vs 9/10),
gpqa 70% (vs 72% — one question), prose **31.2 tok/s** (+8%), TTFT
**0.38–0.41 s** (−15%), 133k prefill 89.4 s (vs ~95), accept 2.32–2.56
(vs 2.26). Pool: 1,324,163 with the drafter (−7.7%: the draft ring-KV
must run bf16 under `--kv-cache-dtype-skip-layers`, where the fp8_e4m3
lane quantizes it; drafterless the lane's pool is 1,858,451, +29.5%).
The quality gains are consistent with group-128 per-token scaling beating
one per-layer scale. Three gotchas earned the hard way:

- **Drafterless + 524k declared fails on EVERY lane**: with ≤4 decode
  rows, `persistent_topk` caps its smem to a small-batch tier and the
  524k row stride then needs 62 CTAs against a 48-CTA budget. The
  drafter's B×8 verify rows land in the 48 KB tier (45 CTAs) — which is
  why production never saw it. Cap drafterless runs at ~358k, or fix the
  kernel heuristic (retry with full smem before failing — upstream-PR
  candidate).
- **`--kv-cache-dtype-skip-layers` list parsing**: a comma-joined index
  string arrives as ONE unmatched element; use the attention-type name
  (`sliding_window` — the DFlash2 draft ring is sliding-window-typed).
- **Acceptance is content-dependent**: an ad-hoc creative-prose probe
  read 1.31/7 accept (18.4 tok/s) on a lane that benches 2.56/7
  (31.2 tok/s) on the standardized c1 phases. Never judge decode from a
  single prompt.

Prefill scales gracefully with depth on this lane (cold, MNBT 8192,
drafter on): 133,186 tokens in 89.4 s (1,490 tok/s) and a **full-bank
499,245 tokens in 390.8 s (1,277 tok/s), needle-exact at that depth** —
a −14% rate falloff from 133k to 499k (indexer scoring and selection
grow with context; no cliff). Budget ~6.5 minutes for a full 500k bank.
Minimum MemAvailable during the full-bank prefill: **3.2 GiB** on the
memory-binding box — tighter than the shipped default's 5.25 GiB floor
(measured at a 112k prefill) but above the 2.26 GiB level §10 rejected;
the full-bank prefill is the floor-binding operation on this lane, and
the strict saturation-plus-full-bank floor methodology is still to run.

The 1M wall (§11) is unchanged — it lives in the indexer's top-k, which
this lane deliberately keeps. The GLM_NEXT lane is also the prerequisite
for NVFP4 KV records (~288 B/token), the next capacity step.

## 13. NVFP4 KV and native 1M (2026-08-30)

Two additions landed together, both riding the GLM_NEXT lane (§12).

**NVFP4 KV (`KV_DTYPE=nvfp4_ds_mla VLLM_NVFP4_MLA_DYNAMIC_SCALE=1`).**
The kernel backport gained a rope-less `(GLM_NEXT, NVFP4_E4M3)` cache
contract: **304 B/token** — 256 B packed E2M1 latent + 32 B E4M3
group-16 scales + one fp32 *per-token* second-level scale (the strict
[0, 288) prefix of the GLM_NSA NVFP4 record, so the latent PTX is
reused unchanged). The dynamic per-token scale needs **no calibration
file** and avoids the shallow-token E4M3-subnormal defect that static
per-layer calibration papers over; static mode is deliberately not
wired (the engine fails closed without the env). Full gauntlet at 524k
vs the fp8_ds_mla default: math_500 **88**/100 (vs 91), gpqa **72%**
(vs 70), estonia **10/10**, lavd **EXACT 10/NEAR 17/FAIL 3**
(vs 15/14/1), prose 30.7 / structured 73.6 tok/s (parity), 133k prefill
95.9 s. **Pool: 1,702,584 tokens = 3.25 × 524k banks (+28.6%).**
Quality sits between the two fp8 lanes — below fp8_ds_mla's
best-in-class lavd, well above fp8_e4m3's.

**Native 1M serves.** The §11 wall was a *launch heuristic*, not
capacity: with ≤8 decode rows `persistent_topk` caps its smem tier and
a 1M row stride needs 90 CTAs against GB10's 48 — so any solo-decoding
request died, at 524k drafterless and at 1M always. The fix retries the
launch computation at the full ~99 KB smem (43 CTAs at 1M) before
failing; on the `v1-dflash2` image it ships as a standalone
`topk_fix.so` (built in-container from `tools/`-adjacent sources)
loaded via `GLM53_TOPK_FIX_SO=/cache/topk_fix.so`; `v2-glmnext` bakes
the fix into `_C`. First native-1M results on this hardware
(`MAX_LEN=1048576` + the NVFP4 lane):

- **Pool 2,144,814 tokens = 2.05 concurrent native-1M banks.**
- Cold 1,029,486-token prefill: **907.8 s (1,134 tok/s)**, needle at
  ~875k depth retrieved EXACT. Depth curve 1,389 tok/s @133k → 1,134
  @1.03M — a gentle −18%, no cliff.
- Decode with ~1.03M resident context: **29.1 tok/s** (empty-context
  baseline 30.7 — flat); warm-prefix follow-up TTFT 19.6 s.
- estonia 10/10; math_500 n=100 **86**/100 (band 86–88 — no
  declaration or topk-fix regression).
- **Floor warning: head MemAvailable bottomed at 0.80 GiB during the
  1M cold prefill** (worker 3.59 GiB) — under the 2.26 GiB line §10
  rejected. Run the 1M profile with `MNBT=4096` for activation
  headroom, or drop the KV budget; 12.4e9 + MNBT 8192 survived but
  with no margin. This is not theoretical: a later concurrency-5
  benchmark sweep on this profile at MNBT 8192 stacked five concurrent
  chunked prefills and **hard-wedged the head box** (kernel-level UMA
  exhaustion, power cycle required). MNBT=4096 is mandatory for any
  concurrent traffic at the 1M declaration.

**Depth-curve completion (cold prefill, tok/s, needle-exact at every
cell):**

| depth  | fp8_ds_mla | nvfp4_ds_mla |
|--------|-----------:|-------------:|
| 133k   | 1,490      | 1,389        |
| 499k   | 1,277      | 1,364        |
| 1.03M  | 1,251      | 1,134        |

fp8 wins shallow, NVFP4 wins mid-depth (the 304 B record moves less
KV per scored token, flattening the curve; crossover ≈200–350k), fp8
retakes the lead at 1M. The default fp8_ds_mla lane also serves the
native-1M declaration with the topk fix: **pool 1,530,144 tokens =
1.46 × 1M banks** — one full 1M bank plus ~480k of spare, where NVFP4
holds 2.05 banks. Floors during the 1M cold prefill: fp8 head
0.85 GiB / worker 3.74 GiB (same MNBT-8192 caveat as above); at 499k
on NVFP4, head 2.49 / worker 5.14 GiB — comfortable.

## 14. spark-arena llama-benchy vs 4-Spark TP4 (2026-08-31)

Full spark-arena methodology (llama-benchy, pp2048/tg128, depths to
131,070, concurrency 1/2/5, 3 runs/cell) plus extra c1/c2 depth rows at
262,144 and 524,288, on the native-1M NVFP4 profile with `MNBT=4096`
and `MAX_SEQS=5`. Reference: leaderboard submission sub1788047509273 —
4× Spark TP4, official FP8 weights, BF16 KV, eager, no speculation.
The ~9.5 h sweep held a 5.37 GiB head-memory floor throughout —
`MNBT=4096` is validated for concurrent traffic at the 1M declaration
(§13's MNBT-8192 wedge does not reproduce at 4096).

| metric (t/s total) | 2× GB10 (this kit) | 4× GB10 TP4 | per-node |
|---|---:|---:|---|
| pp2048 c1 | 962 | 1,104 | **481 vs 276 (+74%)** |
| ctx_pp c1 @131k | 1,355 | 2,185 | **678 vs 546 (+24%)** |
| ctx_pp c1 @262k / @524k | **1,335 / 1,275** | — | they cannot post these |
| ctx_tg c1 (all depths) | 8.4–8.7 flat → **8.42 @524k** | 15.5–15.7 flat → 131k | 4.35 vs 3.88 (+12%) |
| tg128 c1 | 8.80 | 14.50 | 4.40 vs 3.63 |
| ctx_tg @131k c5 | 0.78 | 2.57 | prefill-wall-dominated metric; 2× aggregate prefill wins |

Two readings to keep honest. First, absolute shallow decode goes to
TP4: decode is bandwidth-bound and they have twice the hardware.
Second, the decode rows measure llama-benchy's regime — *sampled*
(server-default temperature) continuation of tiled book text — which
collapses DFlash2 acceptance to 1.14/7 (vs 2.5/7 on prose, measured
from the engine's spec counters over the whole run). The kit's 25–72
tok/s serving numbers come from greedy/structured workloads and are not
comparable to these cells; TP4 runs no speculation and is immune to the
content effect. Our prefill-per-node lead and the 262k/524k rows —
decode still flat at 8.4 with 524k resident — are the structural
results. (A d524288×c5 cell was measured but oversubscribes the KV pool
1.27× and reports admission-queueing, not throughput; excluded.)

Raw table: the run's CSV ships in the results archive; the arena
leaderboard accepts submissions only via its own CLI.

## 15. Concurrency: the sweep, the dflash spec-state wall, and the mixed-prefill decode floor (2026-08-31)

A 16k-context concurrency sweep (c = 1..16, unique-from-token-0
prompts, greedy + ignore_eos, engine-counter deltas) on the GLM_NEXT
default answered three questions:

**The knee is above c=16.** Aggregate throughput still climbs at 16
streams (drafterless c16 = 103.0 tok/s, +17% over c12; MTP-4 c16 =
95.2). Decode never becomes the bind at the concurrency this hardware
can hold — capacity does. `MAX_SEQS=4` at the 524k declaration stands.

**DFlash2 pins ~24–25 KV-pool blocks per running request, independent
of context length.** Speculative decode on the hybrid stack reserves
[k+1 spec-state slots] × [3 mamba cache groups] of full 4608-token
blocks per request for the request's lifetime (plus the running-state
block per group). At k=7 that is ~110k pool tokens per request before
any actual context: a hard ~7-request cap at 16k contexts, a ~3.3×
capacity tax at 50k, and 16 × 24 blocks would exceed the entire 287
block pool — the high-concurrency profile cannot run with DFlash2
today. The fix (dedicated slot pages outside the block pool) requires
a worker-side rewrite of the spec-state indexing and is queued for a
follow-up image; until then choose the speculator by concurrency:
**DFlash2 for c ≤ 4** (best single-stream), **MTP-4 for c ≤ 4 with
tighter memory** (k=4 → ~15 blocks/request), **drafterless for c ≥ 8**
(zero spec-state tax; remember the 358k drafterless cap, §12).

**Mixed prefill starves decode; the launcher now has a floor knob.**
With a 133k cold prefill running next to decode streams, decode
collapses from ~29.5 tok/s (2-stream aggregate) to p50 1.4 tok/s for
the whole prefill — each mixed step carries a full 8k-token chunk with
~5 s step time. The scheduler now supports a mixed-only policy
(`MIXED_PREFILL_CAP`): `-1` skips peer prefill chunks while anything
is decoding, `N` caps them to N tokens, `0`/unset is off (default).
Solo prefills are never touched (unlike
`--long-prefill-token-threshold`, which would destroy solo TTFT), and
an anti-starvation guard (`MIXED_PREFILL_MAX_DEFER`, default 8) forces
one unrestricted step after that many consecutive deferrals so queued
prefills always progress. Measured on the production config:

| setting | decode p50 during 133k prefill | 133k prefill wall |
|---|---:|---:|
| off (default) | 1.4 tok/s (5% of solo) | 89.1 s |
| `MIXED_PREFILL_CAP=-1` (skip) | **11.4 tok/s (39%)** | 125.0 s (1.40×) |
| cap 512 / 1024 / 2048 | 9.2–9.4 tok/s (~31%) | ~126–128 s (1.42×) |

Skip dominates every cap value — per-step overhead, not chunk size,
sets the mixed-step cost (the same shape MiaAI-Lab measured on their
FLASHINFER lane; the guard is this kit's addition). It ships **off by
default** (no arm reached 50% of solo decode, the pre-set adopt bar);
turn it on for interactive/agent serving where decode latency matters
more than cold-prefill TTFT: `MIXED_PREFILL_CAP=-1`. Raising
`MIXED_PREFILL_MAX_DEFER` trades prefill TTFT for a higher floor.

**Agentic profile (documented, not default):** for many concurrent
mid-length agents prefer `MAX_LEN=131072`–`262144`, `MAX_SEQS=12`–`16`,
`MNBT=4096`, `SPEC=none` (or `MTP=4` at the smaller end),
`MIXED_PREFILL_CAP=-1`. The 524k DFlash2 default is tuned for few deep
streams, and the spec-state wall above makes it actively wrong for
high concurrency.

## 16. Fine-grained prefix reuse (v2.1, 2026-09-01)

`--prefix-match-unit 2304` (the v2.1 launcher default) halves the
prefix-cache hash grain below the 4608-token mamba block, waking the
partial-tail machinery: a producer registers its prompt-tail recurrent
state at the last 2304 boundary, and an exact-extension turn resumes
there with only a tail-gap replay. Measured on the 2x-GB10 stack:
warm agentic turns 3.0-3.2 s vs ~12 s cold at 12-17k context (~3.9x
turn rate), accept lengths 7-8/8 on warm turns, factual-QA parity
10/10 warm vs 10/10 cold.

Four engine fixes make this safe (fork commits ad4b9d92a, 3996320a5,
1c66e5df2, 1d220461f — baked in v2.1; the default is ENGINE-FATAL on
v2 and earlier images, whose CoW helper asserts on the first sub-block
partial hit):

- Per-block copy-on-write now skips ring-backed DFlash draft tensors
  (they hold ring-slot pages, not pool pages).
- Drafting survives fine restores: the scheduler reserves one draft
  sliding window (2048 tokens) of replay on every fine hit, and the
  speculator allows block-unaligned restored prefixes once the
  replayed tail covers the draft window (restored garbage falls
  outside every query's window). Without this, warm decode fell to
  dense speed (~10.7 tok/s vs ~28).
- The mamba-align eagle backoff is skipped for DFlash (its draft
  group opts out of prefix caching, so nothing ever eagle-drops):
  previously the LAST complete 4608 boundary was permanently
  unhittable whenever the prompt remainder fit one scheduling chunk -
  this backoff is the entire origin of the historical
  `(complete_blocks - 1) x 4608` warm-hit rule. Every complete block
  boundary is now hittable.

Reuse floors after v2.1: byte-identical repeats and extensions hit
`floor((P - 1 - 2048) / 2304) * 2304`; the producer-tail state at
`floor((P - 1) / 2304) * 2304` becomes reachable once the extension
exceeds ~2048 tokens plus the sub-unit remainder. The 2048-token
reserve is the price of keeping speculative decoding on warm turns
(without it, TTFT drops another ~1.5 s but decode runs drafterless).
`PREFIX_MATCH_UNIT=` (explicit empty) restores the coarse 4608
behavior. Divergent-tail requests (shared doc, different question)
reuse the same boundary states. Interior odd 2304-units have no
stored state by design; block boundaries and producer tails do.

## 17. v2.2: the spec-state ring, fair mixed prefill, state retention, and a right-sized workspace (2026-09-02)

Four engine changes ship in `v2.2-ring` (fork `fdf56c9be`, 13 commits on
`1d220461f`); each was gated live on the reference pair.

**The spec-state ring removes DFlash2's pool tax (§15).** Speculative
decode on the hybrid stack needs k scratch recurrent states per request
per mamba group; the align-mode layout took them as (k+1) pool blocks × 3
groups of 4608-token blocks — ~25 blocks (~110k pool tokens) per running
request, context-independent. v2.2 stamps `spec_state_carveout` on the
mamba KV spec at group build: the scheduler allocates exactly the
drafterless layout (zero spec tail), and the worker splices columns 1..k
of the mamba block table to ring pages carved past the pool inside the
MLA tensors — one page per (slot, spec position, layer), keyed by the
runner's persistent request slot (the pre-copy reads the previous step's
accepted state, so batch-row keying would corrupt). The fused
pre-copy/post-process kernels resolve temporal sources with a positive
token bias through the ring; every destination and the conv shifts stay
table-resolved. The carve is a fixed cost charged inside the KV budget
and scaled by `MAX_SEQS` (≈2.5 GiB with the draft ring at the default 4
slots; ~9 GiB at 16), counted by the memory check at boot.

Receipts (production config unless noted): pool 1,324,163 → 1,051,936 at
the 12.4 GB pin (the carve, as sized) — the 14.4 GB default (below)
brings it to 1,287,194; tail-hit accepts 7.65/7.65 per 8 (= v2.1); 4-way
and 8-way concurrent decodes clean; math_500 n=50 42/50 (band 40–44);
`MAX_SEQS=16` at 131k boots (before, 384 spec blocks exceeded the whole
pool). The §15 sweep completes:

| c | DFlash2 + ring | drafterless | MTP-4 |
|---|---:|---:|---:|
| 4 | **50.9** (accept 2.96) | 47.5 | 48.0 |
| 8 | 58.1 (2.61) | **70.8** | 68.9 |
| 12 | 72.5 (2.52) | **87.8** | 82.1 |
| 16 | 69.3 (2.58) | **103.0** | 95.2 |

(aggregate decode tok/s, unique 16k contexts, greedy, 60 s windows, 0
preemptions; ring arm at `MAX_SEQS=16 MNBT=4096 MAX_LEN=131072
KV_CACHE_MEMORY=18e9`, pool 322k.) The capacity wall is gone — every cell
admits — but the k=7 draft-and-verify step costs ~3.9× a plain decode
step at 16 streams while earning ~2.6× tokens, so DFlash2 peaks at c12
and drafterless keeps winning from c≥8. The ring makes DFlash2 *feasible*
at high concurrency, not preferable: the agentic profile stays
`SPEC=none`. Pin guidance: 16 streams at 131k need ≥16.5e9; sixteen 16k
contexts took 18e9 for a 322k pool.

**Fair mixed prefill: a dynamic time-share gate composed with a
sub-block cap.** §15's skip mode held decode at 39% of solo speed with
~3 s stalls per admitted chunk, because every mixed chunk was a full
4608-token mamba block (the splitter floored sub-block chunks to the
block grid — which is why caps under 4608 measured useless). v2.2 (a)
adds `--mixed-prefill-decode-weight w`: a peer prefill chunk is admitted
once decode-only steps have accumulated w·D/P chunk-walls (w=1 is
processor-sharing fairness across requests; walls are accounted per
completed step in `update_from_output`, because async scheduling
misattributes schedule-interval timing — the naive version measured a
7–14% decode share), and (b) lets a deliberately capped chunk advance
sub-block through the existing private-running-state path, re-aligning at
the next boundary (token-budget crumbs still floor, so boundary states
still materialize). Measured stall quantum: 2.9–3.2 s at 4608 → 1.00 s
at cap 1024 → 0.57 s at cap 512; marginal mixed-prefill service rate
1522 tok/s at 4608 (−4% vs 1591 warm solo) vs ~1160 at both 1024 and 512
(−27%, flat below 1024 — so 512 halves the stall at no extra cost).
Scorecard against constructed ideals (decode: D/(D+1) of solo decode
plus the piggyback iteration inside each mixed step; prefill: warm solo
/ (D+1)), D concurrent decode streams vs one cold 65k prefill:

| arm | D=1 decode / prefill | D=2 | D=3 |
|---|---|---|---|
| `WEIGHT=1.0 CAP=512` | 91% / 63% | 86% / 78% | 85% / 82% |
| `WEIGHT=1.0 CAP=1024` | 85% / 75% | 82% / 102% | 90% / 109% |
| `WEIGHT=1.0` skip mode | 66–83% decode, 118–143% prefill (over-serves prefill) |

Accepts flat 6.8–7.0 throughout; solo prefills untouched; zero cost when
uncontended. Kit recommendation for interactive/agentic serving:
`MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512` (CAP=1024 is the
balanced alternative). Off by default. A variant charging only the
chunk's marginal wall was tested and reverted — it over-corrects under
load (decode 91/86/85% → 83/77/76%).

**State retention: re-age boundary states at retirement.** The
under-load zero of §16's warm hits was queue age, not capacity: align
mode frees interior boundary-state blocks mid-request (LRU-oldest) while
the attention prefix is freed at retirement (young), so churn recycled the
state first and the joint hybrid hit zeroed. v2.2 tracks hash-cached
blocks freed early and moves identity-verified survivors to the LRU-young
end when the request retires (latest boundary first), and fixes a
pin-order bug that made the producer prompt-tail state the request's
first-evicted block. A/B at `KV_CACHE_MEMORY=8e9` (51k doc, 20k
sub-prefix, 6.6k-token fillers): baked v2.1 held 13,824 warm tokens
through 22 fillers then zeroed at 26 (wall 4.9 → 13.8 s); v2.2 held
through 24, stepped down one boundary (9,216; wall 7.8 s) at 26, zero at
28. Exact-extension turns were already protected by touch-and-retire;
the value case is sub-prefix / divergent-tail reuse.

**Right-sized indexer workspace → the 14.4 GB KV default.** The glm5next
sparse indexer sized its prefill K-gather workspace at `max_model_len ×
40` entries (132 B each, 2.58 GiB at 524k) where the deepseek_v4 call
site divides by the compress ratio — found by MiaAI-Lab (#86, credit).
v2.2 bounds it by the legal per-step maximum, min(max_num_seqs, MNBT) ×
cdiv(max_model_len + spec, kpool) = 66 MiB at the production shape, with
a fail-closed assert at the workspace slice; ~2.5 GiB reclaimed per rank.
Receipts (pre-ring): pin 12.4e9 → pool 1,324,163 unchanged; 14.4e9 →
1,559,420 (+17.8%); 14.9e9 → 1,619,915 (+22.3%). Net accounting against
the weeks-stable production footprint: stock + 12.4 ≈ fix + 14.9, so
**14.4e9 nets ~0.5 GiB more headroom than the old pin** and becomes the
default; 14.9e9 is the aggressive option. With the ring's carve inside
the same budget, the v2.2 default pool is 1,287,194 tokens (2.46 × 524k
banks). Read against v2.1's 1,324,163: the idle pool is 2.8% smaller, but
v2.1 also charged ~115k pool tokens per *running* speculative request
(4 streams: ~864k left for context), while v2.2 charges nothing per
request — every running configuration has more context capacity than
before, and only the fully idle prefix-cache capacity gives up 2.8%.

**Fine-grained hashing at 512.** `PREFIX_MATCH_UNIT=512` (from 2304)
puts warm-hit floors on a 512 grid (tail-hit probe 6,912 → 7,168 and
11,520 → 12,288 warm tokens) at no measured hash-CPU cost (cold 65k
prefill 1,658 tok/s, the best recorded), accepts 7.65–7.76/8. Default
since v2.2; the 2048-token draft reserve of §16 still applies.

## 18. Kernel census and two env-only wins (2026-09-02)

A torch-profiler census of the v2.2-ring production configuration (one
request, both ranks, units delimited by the runner's annotations) priced
every kernel family in the 8-row verify step (118 ms GPU: dense bf16
GEMM 39%, trellis MoE 37.5%, drafter 6.8%, NCCL 5%) and the 4,608-token
prefill chunk (2,882 ms: MoE 34%, dense GEMM 20%, NCCL 10.6%, sparse MLA
9%, mHC 10%). Two env-only levers came out of the follow-up A/B window
(production down, peer box idle, same content per arm); one became a
launcher default, the other an opt-in knob:

**b12x for the drafter's MXFP8 GEMMs** (`VLLM_USE_B12X_FP8_GEMM=1`,
opt-in). The DFlash2 drafter's 32 MXFP8 GEMMs per step ran through
FlashInfer's `DeviceGemmMxfp8GemmSm120` at ~65 GB/s effective for M=8
(7.6 ms/step). The fork's MXFP8 kernel list already ranks
`B12xMxfp8LinearKernel` first but gates it behind this env; with it on,
the same 32 launches take 3.1 ms (drafter family 7.7 → 3.2 ms/step,
prefill draft 51 → 21 ms/chunk) and nothing else in the step moves.
Only the drafter uses MXFP8, so target numerics are untouched — but
draft numerics are not: in the same-content acceptance arm
(`sweep_runner --arm t0ab`, 16k contexts, 60 s windows) the b12x head
kept acceptance at 1–2 streams (2.40/2.49 vs 2.42/2.47 accepted per
draft, +4% tok/s from the faster step) and lost it at 4 streams (2.40 ±
0.06 over three runs vs 2.59 ± 0.06 for FlashInfer over four; −8% tok/s
at equal step rate). At 4 streams the drafter GEMMs run M=32 rows, past
b12x's `m <= 8` decode policy, on a different tile path. Until a per-M
dispatch (b12x below ~16 rows, FlashInfer above) lands, the knob stays
off by default.

**Eight fixed NCCL channels** (`NCCL_MIN_NCHANNELS=8
NCCL_MAX_NCHANNELS=8`, launcher default). An all-reduce microbench in the serving image
(two ranks over the 200 GbE rail, launcher transport env) put the
default channel selection at 3.38 ms for the 37.7 MB prefill
all-reduce (11 GB/s), 616 µs at 1 MB and 52 µs at 64 KB. Pinning 8
channels gives 2.96 ms / 160–210 µs / 44–59 µs; 4 channels is a hair
better at 37.7 MB but worse at 8 MB and jittery at 64 KB; QPs per
connection, `NCCL_NCHANNELS_PER_NET_PEER`, `NCCL_BUFFSIZE`, LL128 and
Simple all lose. In-serve: prefill all-reduce 306 → 286 ms/chunk,
decode 6.2 → 5.5 ms/step.

Combined receipt (same boot, both knobs): verify step 118.1 → 111.2 ms
kernel time (−5.2 ms attributable), 131k prefill 82.7 → 81.5 s; the
channel default alone is decode-neutral (step rate 8.66 → 8.66/s at one
stream) and worth ~20 ms per prefill chunk. Rejected in the
same window: `MNBT=9216` (the mamba block alignment keeps prefill
forwards at 4,608 tokens, so the larger workspace only costs: +3% GPU,
92 s wall). Blocked: GPU Direct RDMA over RoCE — every GDR knob reports
`GPU Direct RDMA Disabled` because the image's `libmlx5` lacks
`mlx5dv_reg_dmabuf_mr` and no `nvidia-peermem` is loaded, so 37.7 MB
traffic bounces through host buffers at ~100 Gb/s of the 200 Gb/s link;
a newer rdma-core in the image is the next NCCL lever.

Also measured: the head box runs at 0.5–1.5 GB of available host memory
with the model loaded (gmu 0.85 + 14.4 GB KV). One census was
confounded by memory pressure (NVRM `NV_ERR_NO_MEMORY` in dmesg,
alternating 2× chunk slowdowns per rank); GPU clocks stayed at 2,424
MHz with no throttle flags. If a deployment adds anything to the head,
take the KV budget down a notch (13.4 GB) first.

## Production recommendation

Serve the GLM_NEXT lane (ratified default 2026-08-30):
`ATTN_BACKEND=B12X_MLA_SPARSE KV_DTYPE=fp8_ds_mla
KV_SKIP_LAYERS=sliding_window`, 524k context, DFlash2 k=7, CUDA graphs,
MNBT 8192, KV budget 14.4 GB (v2.2; 12.4 GB on v2.1 and older), prefix
unit 512, gmu 0.85, `SKIP_MM_PROFILING=1`, thinking off at the serving
layer. The `v2.2-ring` image bakes the whole lane (fork @ `fdf56c9be` +
b12x `glm-next-backport`); `v2-glmnext` first baked it, and the older
`v1-dflash2` required hotfix overlays. For interactive or agentic
serving add `MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512`
(§17). The fp8_e4m3 lane (no overlay needed) remains fully
supported: drop the three env knobs. For math-heavy workloads the
GLM_NEXT default now leads (91/100); the short-context bf16 profile
(`MAX_LEN=131072 KV_DTYPE= ATTN_BACKEND= SKIP_MM_PROFILING=0
MAX_SEQS=6`) remains for minimal-quantization preference. `MTP=4` only
as an emergency fallback. Do not raise memory knobs without re-running
the floor methodology.

Two opt-in profiles extend the default (§13): **NVFP4 KV**
(`KV_DTYPE=nvfp4_ds_mla VLLM_NVFP4_MLA_DYNAMIC_SCALE=1`) trades a
little quality headroom (math 88 vs 91, lavd 10 vs 15 EXACT) for +28.6%
pool (3.25 × 524k banks); **native 1M** (`MAX_LEN=1048576` on the NVFP4
lane; on the v1 image add `GLM53_TOPK_FIX_SO=/cache/topk_fix.so`) serves
two concurrent full-length 1M banks — budget ~15 min per cold 1M
prefill and prefer `MNBT=4096` (head-box floor). fp8_ds_mla stays the
default because it wins the quality gates outright.
