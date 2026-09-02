# Community image announcement

Per the community Docker publishing checklist.

**Status:** community derivative (experimental-friendly, production-tested
on one reference kit). Not an official release of any upstream project.

**Image and digest:**
`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.2-ring` (= `:latest`)
digest `sha256:2a2874f0d70e036e7ec47aa635b5ad24d903a0bbf3622b371cd879bfd672ad02`
(previous: `:v2.1-finegrain`
`sha256:50148c7b44e9121470e9735f24fb364fdf591718782fffb6a96c01b5a09bfc16`,
`:v2-glmnext`
`sha256:8fd3892b8a222477678d1e447b8a388c7e2f47c64c4f862ccbc61723501d54e8`,
`:v1-dflash2`
`sha256:284142c5833cbfd540ad42bb8f32cb340451db05a3b84029eaebae54579e9135`)

**Based on:** `nvidia/cuda:13.0.2-base-ubuntu24.04` (aarch64) slim rebase;
JIT toolchain from the `13.0.2-devel` build donor,
via eugr's `spark-vllm-docker` build system
(credited; private — all inputs disclosed in [BUILD.md](BUILD.md)).

**Build recipe:** [BUILD.md](BUILD.md) (build command, baked
build-metadata.yaml, finalize + slim Dockerfiles shipped in-repo; the
shipped b12x IS the base build's b12x — the fork below — so the v1
metadata trap is gone).

**Source commits and PRs:** vLLM branch
[Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark)
@ `fdf56c9be` (public, full history); b12x
[Entrpi/sparkinfer-glmrt](https://github.com/Entrpi/sparkinfer-glmrt)
`glm-next-backport` @ `3ce6115`; exllamav3 `c5d9c657`; FlashInfer 0.6.18
`083012d6` + 2 sm12x patches (shipped). No unlisted patches.

**Changes from base:** README "Under the hood" — glm5_next port + sm121
fixes, EXL3 standard-checkpoint fused serving, DFlash2 transplant + salted
Gumbel + non-causal draft attention, mHC aux capture, draft-KV grouping +
ring draft KV, chat-template repairs (multimodal + enable_thinking gating),
FlashInfer fp8-MLA sm12x patches. Since v2: the GLM_NEXT b12x lane as the
baked default (`B12X_MLA_SPARSE` + `fp8_ds_mla` packed KV), the
instanttensor direct-I/O loader, mixed-prefill decode-floor knobs. Since
v2.1: fine-grained prefix reuse (ring-safe per-block copy-on-write, a
draft-replay reserve + gap-aware restore guard, the mamba eagle-backoff
fix). New in v2.2: the **mamba spec-state ring** — DFlash2/MTP scratch
recurrent states leave the KV pool for per-slot ring pages carved from the
MLA tensors, removing the ~25-block-per-request pool tax that capped
DFlash2 at 7 concurrent 16k streams (it now boots at 16); a **fair
mixed-prefill scheduler** (dynamic time-share gate composed with a
sub-block chunk cap: decode keeps 85–91% of its processor-sharing share
during cold prefills with 0.57 s stalls, vs 39% / ~3 s in the v2 skip
mode); **boundary-state retention** (early-freed hashed states are re-aged
at retirement, so sub-prefix reuse degrades from the tail instead of
zeroing under churn); a **right-sized sparse-indexer prefill workspace**
(the call-site asymmetry found by MiaAI-Lab's #86 — ~2.5 GiB reclaimed per
rank at 524k, funding the new 14.4 GB KV default); and
`--prefix-match-unit 512` as the fine-hash default. Inherited: b12x
runtime, breakable CUDA graphs, kpool sparse attention
(local-inference-lab lineage).

**Tested configuration:** 2x GB10 (DGX Spark), TP2 over 200GbE RoCE,
brandonmusic EXL3/TR3 4bpw (120 shards), `--quantization exl3`, DFlash2
`num_speculative_tokens=7` (MTP-4 fallback), CUDA graphs on,
`gpu-memory-utilization 0.85`, `kv-cache-memory 14.4e9` (12.4e9 on v2.1
and older), MNBT 8192, NFS and local-weights topologies. Shipping default
(the GLM_NEXT lane, ratified 2026-08-30): `--max-model-len 524288`,
`fp8_ds_mla` packed KV + `B12X_MLA_SPARSE` (sliding-window layers
excepted), `--max-num-seqs 4`, instanttensor loader (~3.6 min
launch-to-API), fine-grained prefix hashing at 512, spec-state ring on —
1,287,194-token pool (2.46 x 524k banks). Opt-in profiles: NVFP4 KV
(+28.6% pool), native 1M (2.05 concurrent full banks) per FINDINGS section
13, and the agentic profile (`MAX_SEQS` 12–16, `SPEC=none`,
`MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512`) per section 17.

**Validation results:** 20/20 self-containment checks
(`scripts/check_image.sh`) and the unit suites in the pulled image
(80 tests incl. the mamba chunk-split, share-gate and retention
suites); on the shipping default: math_500 91% (n=100) and 40 + 43 of 50
on the v2.2 bake run (v2.1 receipts 44 and 40+43 of 50), gpqa_diamond 70%
(n=50), estonia 133k retrieval 10/10, lavd ledger audit 15/30 EXACT,
structured-output 10/10 under the drafter; speculative equivalence
lossless-up-to-ties (`tools/dflash_equiv.py`); c1 ~28.5-29.5 tok/s prose,
TTFT ~0.47 s. v2.2 receipts: tail-hit warm accepts 7.65/8 with the ring
(= v2.1), 4- and 8-way concurrent decodes clean, DFlash2 concurrency sweep
complete at 16 streams (50.9/58.1/72.5/69.3 tok/s at c4/8/12/16; the
capacity wall is gone, drafterless still wins from c8), fairness scorecard
91/86/85% of ideal decode at 1–3 streams under a cold 65k prefill,
retention A/B graceful step-down vs hard zero (FINDINGS section 17,
reproduction scripts in-repo).

**Known limitations:** `--prefix-match-unit` below 4608 is ENGINE-FATAL on
`v2-glmnext` and older images (the fixes are in v2.1+'s vLLM) — the
launcher only defaults it with the matching image; the 14.4 GB KV default
assumes v2.2's right-sized indexer workspace — pin `KV_CACHE_MEMORY=12.4e9`
on older images; the spec-state ring's fixed cost scales with
`--max-num-seqs` and is charged inside the KV budget (16 streams at 131k
need >= 16.5e9; the `_check_enough_kv_cache_memory` error at boot is the
tell; `VLLM_MAMBA_SPEC_CARVEOUT=0` restores pool-block spec state); video
inputs untested; long-context declarations past 524,288 on the default
lane need the NVFP4 profile (native-1M is validated there); drafterless
runs on v1 images should cap `--max-model-len` at ~358k; draft TP=1 knob is
plumbed but inert on this fork; interior 512-unit states between block
boundaries are by design not retained (block boundaries and prompt tails
are), and warm turns pay a fixed 2,048-token replay reserve to keep the
drafter active.

**Support contact or issue tracker:**
[github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues)
(maintainer: Entrpi). Debug against this derivative before assigning
problems upstream.

**Upstreaming:** focused PRs against the local-inference-lab fork are
planned, strongest first: the mixed-prefill share gate + chunk cap (the
generalization of both the decode floor and a prefill schedule interval),
the mamba eagle-backoff fix + fine-restore draft guard (the v2.1
mechanism), boundary-state retention, the indexer workspace call-site
asymmetry (with the deepseek_v4 precedent and MiaAI-Lab credit), the
spec-state ring, FlashKDA-for-GLM, eagle draft-group annotation, ring
draft-KV + glm5 draft grouping, the sparkinfer b12x fixes to their lineage
repos, and the sm121 `persistent_topk` retry.
