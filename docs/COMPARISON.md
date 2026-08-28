# Cross-stack: this recipe vs MiaAI-Lab's

[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
serves the same model, the same brandonmusic EXL3 4bpw checkpoint, and the
same incoai DFlash2 k=7 drafter on the same class of hardware — through an
almost completely different stack. Both projects were developed
independently and compared after the fact. This document is the honest
side-by-side.

## Different roads

| | This recipe | MiaAI recipe |
|---|---|---|
| vLLM base | local-inference-lab fork branch, image rebuilt from source | public day-0-family image pinned by digest + anchored patch scripts |
| Attention | fork's NoPE-MLA lane, DeepGEMM kpool, block 2304 | `FLASHINFER_MLA_SPARSE_SM120`, NoPE latent zero-padded into GLM_NSA 576 |
| KV format | **bf16 default** (fp8_e4m3 gated option) | packed `fp8_ds_mla` **mandatory** (bf16 has no sparse kernel there) |
| Context / pool | 131k, 520,470 tok (3.97×); fp8 option 769,817 | **900k**, 982,612 tok (1.09× — one full request) |
| Prefill chunk | 8192 (112k prefill ~55 s) | 1024 (8192 oversubscribes their indexer top-k) |
| Fused EXL3 MoE | tpurtell b12x CuTeDSL trellis | turboderp `exllamav3_ext.exl3_moe` |
| Draft placement | TP2 (sharded with the target) | TP1 on rank 0 |
| Spec fallback | MTP-4 (28.6 tok/s) | MTP-2 (~24.6 tok/s) |
| Quality evidence | task evals (math/gpqa/vision) + equivalence protocol | weights-level KLD panel (cited) |

## Convergent engineering

Three problems have no upstream solution, and both projects independently
solved all three:

1. **Drafter KV on the glm5 hybrid.** Both found that the drafter's plain
   sliding-window specs eject the model from the glm5 KV fast path onto a
   generic path that cannot serve it. Ours: dedicated draft groups + a
   fixed ring of draft pages with a positional remap kernel. Theirs: an
   exact-fit block rescale that slot-shares drafter layers into the MLA
   tensors. Both end at ~zero pool cost.
2. **mHC aux capture.** Identical contract in both: materialize the
   deferred `hc_post`, then `hc_contract`, at tap layers (6,15,25,34,43).
3. **Non-causal draft attention.** Both discovered (independently, on
   different backends) that a causal mask inside the draft block silently
   collapses later-position acceptance.

Independent replication of the same three fixes is strong evidence both
stacks implement DFlash2-on-GLM-5.3 correctly.

## Numbers, on their protocol

Their `bench_decode.py` vendored unmodified
([scripts/bench_decode_miaai.py](../scripts/bench_decode_miaai.py)); their
side from their published README (same protocol, their hardware):

| | This stack | MiaAI |
|---|---:|---:|
| Structured (count 1→200) tok/s | **72.4** (66.7–68.9 pre-template-fix) | 61.7 lab / 62.9 official |
| — accept/step of 7 | 6.86 | 6.43 |
| Prose (hash-map) tok/s | 27.4 | 26.9 |
| — accept/step | 2.17 | 2.33 |
| TTFT short prompt | 0.43–0.47 s | ~0.72 s |
| MTP baseline | 28.6 (k=4) | ~24.6 (k=2) |

Reading: per engine step the stacks are near-identical (~115 ms) — two
different fused-MoE engines landing at par. The structured gap is mostly
acceptance (their draft accepts slightly better on prose, ours on
structured post-template-fix); the TTFT gap favors this stack consistently.

## What each stack is for

- **Choose the MiaAI recipe** for maximum context: 900k tokens on two
  Sparks is a real capability this recipe does not attempt (its packed-fp8
  KV is ~40% smaller per token, and mandatory).
- **Choose this recipe** for production serving at 131k: bf16-KV quality
  defaults with n≥50 task gates, 4× concurrency, ~8× faster long-prompt
  prefill, deeper fallbacks, and the acceptance/memory methodology to
  extend it safely.

Ideas already carried over from their recipe, with credit: JIT cache
persistence, post-boot shape warmup, and the `enable_thinking` template
gating (which fixed a real quality bug here and added +7% structured
throughput). Their draft-TP=1 idea is noted but inert on this fork (the
dflash lane ignores draft parallel config; porting it is parked).
