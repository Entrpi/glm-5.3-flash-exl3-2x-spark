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

## Production recommendation

Serve the defaults: 524k context, fp8 KV, DFlash2 k=7, CUDA graphs,
MNBT 8192, KV budget 12.4 GB, gmu 0.85, `SKIP_MM_PROFILING=1`, thinking
off at the serving layer. For math-heavy workloads use the short-context
profile (`MAX_LEN=131072 KV_DTYPE= SKIP_MM_PROFILING=0 MAX_SEQS=6` — bf16,
94% math_500). `MTP=4` only as an emergency fallback. Do not raise memory
knobs without re-running the floor methodology.
