# Image provenance: ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark

**Status: community derivative.** Not an official image of vLLM,
local-inference-lab, eugr, or NVIDIA.

## Identity

| | |
|---|---|
| Tag | `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v1-dflash2` |
| Digest | `sha256:284142c5833cbfd540ad42bb8f32cb340451db05a3b84029eaebae54579e9135` (`:v1-dflash2` = `:latest`) |
| Base image | `nvidia/cuda:13.0.2-devel-ubuntu24.04` (aarch64), manifest-list digest `sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6` |
| Arch | linux/arm64, CUDA kernels compiled for `12.1a` (GB10 / sm_121) |

## How it was built

Two layers:

1. **Base build** — eugr's `spark-vllm-docker` build system (credited;
   currently a private repository — every input it consumes is listed here
   so the image is inspectable and comparable even though the orchestration
   is not one-command reproducible from this repo alone):

   ```
   ./build-and-copy.sh -t vllm-node-glm53 --exp-b12x \
       --vllm-source-dir <clone of Entrpi/vllm-glm-5.3-flash-spark> --rebuild-vllm
   ```

   The build bakes `/workspace/build-metadata.yaml` into the image
   (inspect it: `docker run --rm --entrypoint cat <image> /workspace/build-metadata.yaml`):

   ```yaml
   build_date: 2026-08-28T16:21:14Z
   vllm_version: 0.26.1rc1.dev960+g90104cfe4.d20260828
   vllm_commit: 90104cfe4fd67e93e90ae5afadc0f011ebc06fb8   # = the public branch HEAD
   flashinfer_commit: 083012d6
   gpu_arch: 12.1a
   build_args:
     torch_version: "2.13.0"
     torchvision_version: "0.28.0"
     torchaudio_version: "2.11.0"
     cutlass_dsl_version: "4.7.0"
     b12x_repo: "https://github.com/lukealonso/b12x.git"   # SEE WARNING BELOW
     b12x_ref: "master"
     transformers_5: true
   ```

2. **Finalize layer** — [`Dockerfile.phase3-finalize`](Dockerfile.phase3-finalize)
   (shipped in this directory, verbatim), which bakes:
   - `exllamav3_ext` compiled for sm121 (torch 2.13.0+cu130 ABI) at
     `/exl3ext`, from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) @ `c5d9c657`
   - **b12x replaced wholesale** with
     [tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt)
     `b12x/` @ `fefb9c5` (Apache-2.0)
   - Two FlashInfer sm12x patches over the 0.6.18 install (fp8-MLA
     capability gate in `mla/_core.py`; `CTA_TILE_KV` cap in `mla.cuh`) —
     required only for the `KV_DTYPE=fp8_e4m3` option; live-validated

> **Metadata trap:** the baked `build-metadata.yaml` records the BASE
> build's b12x (`lukealonso/b12x @ master`). The shipped image's b12x is
> NOT that — the finalize layer replaces it with sparkinfer-glmrt
> `fefb9c5`. Anyone reproducing from the metadata alone would build a
> different (non-working, for this EXL3 lane) image. This file is
> authoritative over the baked metadata.

## Source commits

| Component | Source | Ref |
|---|---|---|
| vLLM | [Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark) | `90104cfe4` (branch `main`, full history incl. the local-inference-lab fork lineage) |
| b12x | [tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt) | `fefb9c5` |
| exllamav3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | `c5d9c657` |
| FlashInfer | prebuilt 0.6.18 (`083012d6`) + the two sm12x patches above | — |
| torch / vision / audio | 2.13.0 / 0.28.0 / 2.11.0 (cu130, aarch64) | — |
| CUTLASS DSL | 4.7.0 | — |

Changes from base vs inherited functionality: see the "Under the hood"
table in the [README](../README.md) — everything above the fork lineage
line is inherited; the table rows are what this branch introduces.

## Verification

`scripts/check_image.sh` asserts the image is self-contained (15 checks:
branch version stamp, DFlash2/ring code, chat templates incl. the thinking
gate, exl3ext, b12x tree, FlashInfer patches). `scripts/run_tests.sh` runs
the 41-test suite inside the pulled image. Serving-level validation
commands and results: README "Benchmarks" + "Reproducing".

## Support

Maintainer: [Entrpi](https://github.com/Entrpi). Issues:
[this repo's tracker](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues).
Please debug against this derivative (and read this file) before assigning
a problem to vLLM, local-inference-lab, eugr, or FlashInfer upstream.
