# Community image announcement

Per the community Docker publishing checklist.

**Status:** community derivative (experimental-friendly, production-tested
on one reference kit). Not an official release of any upstream project.

**Image and digest:**
`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v1-dflash2`
digest `sha256:284142c5833cbfd540ad42bb8f32cb340451db05a3b84029eaebae54579e9135`

**Based on:** `nvidia/cuda:13.0.2-devel-ubuntu24.04` (aarch64), manifest-list
digest `sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6`,
via eugr's `spark-vllm-docker` build system
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
brandonmusic EXL3/TR3 4bpw (120 shards), `--quantization exl3`, DFlash2
`num_speculative_tokens=7` (MTP-4 fallback), CUDA graphs on,
`gpu-memory-utilization 0.85`, `kv-cache-memory 12.4e9`, MNBT 8192, NFS
and local-weights topologies (NFS is the production-validated one).
Default serving profile (2026-08-29): `--max-model-len 524288`,
`fp8_e4m3` KV, `--max-num-seqs 4` — 1,435,070-token pool, estonia
133k-retrieval 9/10, lavd n=30 parity vs bf16. Short-context profile:
131,072 / bf16 KV / seqs 6 (block 2304).

**Validation results:** 41/41 unit tests in the pulled image
(`scripts/run_tests.sh`); 15/15 self-containment checks
(`scripts/check_image.sh`); math_500 94% (n=50), gpqa_diamond 78% (n=50),
fp8-KV option gate 89% (n=100); speculative equivalence lossless-up-to-ties
(`tools/dflash_equiv.py`); vision + tool-call smoke at temp 0; c1 33–35
tok/s prose / 72 structured, TTFT 0.42–0.51 s (README "Benchmarks",
commands in "Reproducing").

**Known limitations:** local-both weights topology carries a documented
head swap-wedge risk during load (`--nfs` is the validated fallback); video
inputs untested; 900k single-request context not attempted (see the MiaAI
recipe; this recipe's default is 524k banks × 2.74 concurrency); draft
TP=1 knob is plumbed but inert on this fork; the MXFP8-quantized DFlash2
draft checkpoint does not load **in this image build** (the loader fix is
on the branch @ `88ca596c6` and ships in the next image — until then use
the bf16 drafter with this image).

**Support contact or issue tracker:**
[github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues)
(maintainer: Entrpi). Debug against this derivative before assigning
problems upstream.

**Upstreaming:** focused PRs against the local-inference-lab fork are
planned for the generally useful pieces: the ring draft-KV + glm5 draft
grouping, the mask_embedding loader fix, and the chat-template repairs.
