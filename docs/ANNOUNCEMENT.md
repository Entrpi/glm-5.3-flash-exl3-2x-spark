# Community image announcement

Per the community Docker publishing checklist.

**Status:** community derivative (experimental-friendly, production-tested
on one reference kit). Not an official release of any upstream project.

**Image and digest:**
`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v1-dflash2`
digest `TODO-FILLED-AT-PUSH`

**Based on:** `nvidia/cuda:13.0.2-devel-ubuntu24.04` (aarch64), digest
`TODO-FILLED-AT-PUSH`, via eugr's `spark-vllm-docker` build system
(credited; private — all inputs disclosed in [BUILD.md](BUILD.md)).

**Build recipe:** [BUILD.md](BUILD.md) (build command, baked
build-metadata.yaml verbatim, finalize Dockerfile shipped in-repo, and the
one metadata trap called out: the shipped b12x is sparkinfer-glmrt
`fefb9c5`, not the base build's lukealonso master).

**Source commits and PRs:** vLLM branch
[Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark)
@ `90104cfe4` (public, full history); b12x sparkinfer-glmrt `fefb9c5`;
exllamav3 `c5d9c657`; FlashInfer 0.6.18 `083012d6` + 2 sm12x patches
(shipped). No unlisted patches.

**Changes from base:** README "Under the hood" — glm5_next port + sm121
fixes, EXL3 standard-checkpoint fused serving, DFlash2 transplant + salted
Gumbel + non-causal draft attention, mHC aux capture, draft-KV grouping +
ring draft KV, chat-template repairs (multimodal + enable_thinking gating),
FlashInfer fp8-MLA sm12x patches. Inherited: b12x runtime, breakable CUDA
graphs, kpool sparse attention (local-inference-lab lineage).

**Tested configuration:** 2× GB10 (DGX Spark), TP2 over 200GbE RoCE,
brandonmusic EXL3/TR3 4bpw (120 shards), `--quantization exl3`, bf16 KV
(fp8_e4m3 gated option), DFlash2 `num_speculative_tokens=7`
(MTP-4 fallback), CUDA graphs on, `--max-model-len 131072`,
`--max-num-seqs 6 --max-num-batched-tokens 8192 --block-size 2304`,
`gpu-memory-utilization 0.85`, `kv-cache-memory 12.4e9`, NFS and
local-weights topologies (NFS is the production-validated one).

**Validation results:** 41/41 unit tests in the pulled image
(`scripts/run_tests.sh`); 15/15 self-containment checks
(`scripts/check_image.sh`); math_500 94% (n=50), gpqa_diamond 78% (n=50),
fp8-KV option gate 89% (n=100); speculative equivalence lossless-up-to-ties
(`tools/dflash_equiv.py`); vision + tool-call smoke at temp 0; c1 33–35
tok/s prose / 72 structured, TTFT 0.42–0.51 s (README "Benchmarks",
commands in "Reproducing").

**Known limitations:** local-both weights topology carries a documented
head swap-wedge risk during load (`--nfs` is the validated fallback); video
inputs untested; 900k-context serving not attempted (see the MiaAI recipe);
`KV_DTYPE=fp8_e4m3` is an option, not the default; draft TP=1 knob is
plumbed but inert on this fork.

**Support contact or issue tracker:**
[github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues)
(maintainer: Entrpi). Debug against this derivative before assigning
problems upstream.

**Upstreaming:** focused PRs against the local-inference-lab fork are
planned for the generally useful pieces: the ring draft-KV + glm5 draft
grouping, the mask_embedding loader fix, and the chat-template repairs.
