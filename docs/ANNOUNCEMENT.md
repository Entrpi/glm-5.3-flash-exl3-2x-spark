# Community image announcement

Per the community Docker publishing checklist.

**Status:** community derivative (experimental-friendly, production-tested
on one reference kit). Not an official release of any upstream project.

**Image and digest:**
`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.1-finegrain` (= `:latest`)
digest `sha256:50148c7b44e9121470e9735f24fb364fdf591718782fffb6a96c01b5a09bfc16`
(previous: `:v2-glmnext`
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
@ `1d220461f` (public, full history); b12x
[Entrpi/sparkinfer-glmrt](https://github.com/Entrpi/sparkinfer-glmrt)
`glm-next-backport` @ `3ce6115`; exllamav3 `c5d9c657`; FlashInfer 0.6.18
`083012d6` + 2 sm12x patches (shipped). No unlisted patches.

**Changes from base:** README "Under the hood" — glm5_next port + sm121
fixes, EXL3 standard-checkpoint fused serving, DFlash2 transplant + salted
Gumbel + non-causal draft attention, mHC aux capture, draft-KV grouping +
ring draft KV, chat-template repairs (multimodal + enable_thinking gating),
FlashInfer fp8-MLA sm12x patches. New since v2: the GLM_NEXT b12x lane as
the baked default (`B12X_MLA_SPARSE` + `fp8_ds_mla` packed KV), the
instanttensor direct-I/O loader, mixed-prefill decode-floor knobs.
New in v2.1: fine-grained prefix reuse (`--prefix-match-unit 2304`
default) — ring-safe per-block copy-on-write, a draft-replay reserve +
gap-aware restore guard so speculative decoding survives warm hits, and
a mamba eagle-backoff fix that makes every complete 4,608-token
boundary state hittable (the old `(complete blocks - 1) x 4608` reuse
ceiling was this bug). Inherited: b12x runtime, breakable CUDA graphs,
kpool sparse attention (local-inference-lab lineage).

**Tested configuration:** 2x GB10 (DGX Spark), TP2 over 200GbE RoCE,
brandonmusic EXL3/TR3 4bpw (120 shards), `--quantization exl3`, DFlash2
`num_speculative_tokens=7` (MTP-4 fallback), CUDA graphs on,
`gpu-memory-utilization 0.85`, `kv-cache-memory 12.4e9`, MNBT 8192, NFS
and local-weights topologies. Shipping default (the GLM_NEXT lane,
ratified 2026-08-30): `--max-model-len 524288`, `fp8_ds_mla` packed KV
+ `B12X_MLA_SPARSE` (sliding-window layers excepted), `--max-num-seqs
4`, instanttensor loader (~3.6 min launch-to-API), fine-grained prefix
hashing on — 1,324,163-token pool (2.53x 524k banks; 1,858,451
drafterless). Opt-in profiles: NVFP4 KV (+28.6% pool) and native 1M
(2.05 concurrent full banks) per FINDINGS section 13.

**Validation results:** 15/15 self-containment checks
(`scripts/check_image.sh`) and the unit suites in the pulled image
(incl. the new mamba chunk-split suite, 21/21); on the shipping
default: math_500 91% (n=100) / 44 and 40+43 of 50 across three n=50
bake runs, gpqa_diamond 70% (n=50), estonia 133k retrieval 10/10, lavd
ledger audit 15/30 EXACT, structured-output 10/10 under the drafter;
speculative equivalence lossless-up-to-ties (`tools/dflash_equiv.py`);
c1 ~28.5-29.5 tok/s prose, TTFT ~0.47 s. Fine-grain receipts (v2.1):
warm agentic turns 3.0-3.2 s vs ~12 s cold at 12-17k context,
factual-QA warm/cold parity 10/10 vs 10/10, speculative accepts 7-8/8
on warm turns including block-unaligned tail restores (FINDINGS
section 16, reproduction scripts in-repo).

**Known limitations:** `--prefix-match-unit 2304` is ENGINE-FATAL on
`v2-glmnext` and older images (the fixes are in v2.1's vLLM) — the
launcher only defaults it with the matching image; video inputs
untested; long-context declarations past 524,288 on the default lane
need the NVFP4 profile (native-1M is validated there; the
`persistent_topk` small-batch retry that makes drafterless 524k boot
is baked since v2); drafterless runs on v1 images should cap
`--max-model-len` at ~358k; draft TP=1 knob is plumbed but inert on
this fork; interior 2,304-unit states between block boundaries are by
design not retained (block boundaries and prompt tails are), and warm
turns pay a fixed 2,048-token replay reserve to keep the drafter
active.

**Support contact or issue tracker:**
[github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues)
(maintainer: Entrpi). Debug against this derivative before assigning
problems upstream.

**Upstreaming:** focused PRs against the local-inference-lab fork are
planned, strongest first: the mixed-prefill decode floor + starvation
guard, the mamba eagle-backoff fix + fine-restore draft guard (the
v2.1 mechanism), FlashKDA-for-GLM, eagle draft-group annotation, ring
draft-KV + glm5 draft grouping, the sparkinfer b12x fixes to their
lineage repos, and the sm121 `persistent_topk` retry.
