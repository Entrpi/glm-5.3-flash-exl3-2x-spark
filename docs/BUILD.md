# Image provenance: ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark

**Status: community derivative.** Not an official image of vLLM,
local-inference-lab, eugr, or NVIDIA.

## Identity (current: v2.3-tier1)

| | |
|---|---|
| Tag | `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1` |
| Digest | `sha256:44b2dbafe8d81d6c7019a346480b49b23c1a0bef99c9b1d0db6e7ccc926921ca` (`:v2.3-tier1` = `:latest`) |
| Base image | `nvidia/cuda:13.0.2-base-ubuntu24.04` (aarch64; slim rebase — the JIT toolchain is copied from the devel build donor) |
| Arch | linux/arm64, CUDA kernels compiled for `12.1a` (GB10 / sm_121) |

Delta vs `v2.2-ring`: vLLM `f223ff9f2` (5 commits — the rowwise-fp8
DFlash2 draft head, the per-M dispatch of the b12x MXFP8 GEMM with its
FlashInfer fallback, their unit tests) and b12x `2f53ce3` (the W4A16
route-pack fast path); all Python/Triton, same 3-layer build, same
FlashInfer 0.6.18 patches. Launcher defaults move to
`NCCL_MIN/MAX_NCHANNELS=8`, `VLLM_USE_B12X_FP8_GEMM=1`,
`VLLM_DFLASH_FP8_DRAFT_HEAD=1` (FINDINGS §18–19).

Previous: `:v2.2-ring`
`sha256:2a2874f0d70e036e7ec47aa635b5ad24d903a0bbf3622b371cd879bfd672ad02`
(vLLM `fdf56c9be`). Its delta vs `v2.1-finegrain`: 13 commits — the mamba
spec-state ring (DFlash2/MTP scratch states move off the KV pool onto
per-slot ring pages carved from the MLA tensors; the ~25-block-per-request
pool tax is gone), the dynamic time-share mixed-prefill gate composed with
a sub-block chunk cap, boundary-state retention (re-age at retirement),
the indexer prefill-workspace right-size (MiaAI-Lab #86), the FWHT
query-scale fusion and physical-storage pool-expansion riders; pure
Python incl. Triton kernel edits — no CUDA rebuild). Same 3-layer build,
same b12x `3ce6115`, same FlashInfer 0.6.18 patches.

Previous: `:v2.1-finegrain`
`sha256:50148c7b44e9121470e9735f24fb364fdf591718782fffb6a96c01b5a09bfc16`
(vLLM `1d220461f` — fine-grained prefix reuse; pin
`KV_CACHE_MEMORY=12400000000` on it, its indexer workspace is not
right-sized), `:v2-glmnext`
`sha256:8fd3892b8a222477678d1e447b8a388c7e2f47c64c4f862ccbc61723501d54e8`
(vLLM `c83d60a5b`; the fine-grained default `PREFIX_MATCH_UNIT=2304` is
ENGINE-FATAL on it — use `PREFIX_MATCH_UNIT=` there), and `:v1-dflash2`
`sha256:284142c5833cbfd540ad42bb8f32cb340451db05a3b84029eaebae54579e9135`
(base `nvidia/cuda:13.0.2-devel-ubuntu24.04`,
`sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6`) —
still pullable; the GLM_NEXT lane on it requires the hotfix overlays that
v2 bakes.

## How it was built

Three layers:

1. **Base build** — eugr's `spark-vllm-docker` build system (credited;
   currently a private repository — every input it consumes is listed here
   so the image is inspectable and comparable even though the orchestration
   is not one-command reproducible from this repo alone):

   ```
   ./build-and-copy.sh -t vllm-node-glm53 --exp-b12x \
       --vllm-source-dir <clone of Entrpi/vllm-glm-5.3-flash-spark @ f223ff9f2> --rebuild-vllm
   ```

   with the harness's b12x source pinned to
   `Entrpi/sparkinfer-glmrt` branch `glm-next-backport` (@ `2f53ce3`) —
   unlike v1, the shipped b12x now IS the base build's b12x (the v1
   "metadata trap" is gone). The build bakes
   `/workspace/build-metadata.yaml` into the image
   (inspect: `docker run --rm --entrypoint cat <image> /workspace/build-metadata.yaml`).

2. **Finalize layer** — [`Dockerfile.phase3-finalize`](Dockerfile.phase3-finalize)
   (the v2 version — its v1 predecessor also carried a b12x
   whole-tree COPY that v2 drops, see the file header), which bakes:
   - `exllamav3_ext` compiled for sm121 (torch 2.13.0+cu130 ABI) at
     `/exl3ext`, from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) @ `c5d9c657`
   - Two FlashInfer sm12x patches over the 0.6.18 install (fp8-MLA
     capability gate in `mla/_core.py`; `CTA_TILE_KV` cap in `mla.cuh`) —
     required only for the `KV_DTYPE=fp8_e4m3` option; live-validated

3. **Slim rebase** — [`Dockerfile.slim`](Dockerfile.slim): the python
   userland + JIT toolchain rebased onto `cuda:13.0.2-base` (drops the
   devel toolkit's static archives and duplicated .so; the pip nvidia
   wheels torch actually loads are kept and wired via ld.so.conf).
   Technique credit: AEON-7/vllm-ultimate-dgx-spark.

## Source commits

| Component | Source | Ref |
|---|---|---|
| vLLM | [Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark) | `f223ff9f2` (the image's version stamp = branch `main`; v2.2-ring was `fdf56c9be`; full history incl. the local-inference-lab fork lineage) |
| b12x | [Entrpi/sparkinfer-glmrt](https://github.com/Entrpi/sparkinfer-glmrt) | branch `glm-next-backport` @ `2f53ce3` (route-pack fast path on top of `3ce6115`; fork of tpurtell/sparkinfer-glmrt, lukealonso/b12x lineage) |
| exllamav3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | `c5d9c657` |
| FlashInfer | prebuilt 0.6.18 (`083012d6`) + the two sm12x patches above | — |
| torch / vision / audio | 2.13.0 / 0.28.0 / 2.11.0 (cu130, aarch64) | — |
| CUTLASS DSL | 4.7.0 | — |

Changes from base vs inherited functionality: see the "Under the hood"
table in the [README](../README.md) — everything above the fork lineage
line is inherited; the table rows are what this branch introduces.

## Verification

`scripts/check_image.sh` asserts the image is self-contained (20 checks:
branch version stamp, DFlash2/ring code, the v2.2 spec-state ring, share
gate + chunk cap, retention and indexer-workspace markers, chat templates
incl. the thinking gate, exl3ext, b12x tree, FlashInfer patches).
`scripts/run_tests.sh` runs the 87-test suite inside the pulled
image. Serving-level validation
commands and results: README "Benchmarks" + "Reproducing".

## Support

Maintainer: [Entrpi](https://github.com/Entrpi). Issues:
[this repo's tracker](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues).
Please debug against this derivative (and read this file) before assigning
a problem to vLLM, local-inference-lab, eugr, or FlashInfer upstream.
