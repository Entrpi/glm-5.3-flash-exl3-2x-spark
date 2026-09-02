# Community image announcement

Per the community Docker publishing checklist.

**Status:** community derivative (experimental-friendly, production-tested
on one reference pair). Not an official release of any upstream project.

**Image and digest:**
`ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1` (= `:latest`)
digest `sha256:44b2dbafe8d81d6c7019a346480b49b23c1a0bef99c9b1d0db6e7ccc926921ca`
(previous: `:v2.2-ring`
`sha256:2a2874f0d70e036e7ec47aa635b5ad24d903a0bbf3622b371cd879bfd672ad02`,
`:v2.1-finegrain`
`sha256:50148c7b44e9121470e9735f24fb364fdf591718782fffb6a96c01b5a09bfc16`,
`:v2-glmnext`
`sha256:8fd3892b8a222477678d1e447b8a388c7e2f47c64c4f862ccbc61723501d54e8`,
`:v1-dflash2`
`sha256:284142c5833cbfd540ad42bb8f32cb340451db05a3b84029eaebae54579e9135`)

**Based on:** `nvidia/cuda:13.0.2-base-ubuntu24.04` (aarch64) slim rebase;
JIT toolchain from the `13.0.2-devel` build donor, via eugr's
`spark-vllm-docker` build system (credited; private — all inputs disclosed
in [BUILD.md](BUILD.md)).

**Build recipe:** [BUILD.md](BUILD.md) (build command, baked
build-metadata.yaml, finalize + slim Dockerfiles shipped in-repo; the
shipped b12x is the base build's b12x — the fork below).

**Source commits and PRs:** vLLM branch
[Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark)
@ `f223ff9f2` (public, full history); b12x
[Entrpi/sparkinfer-glmrt](https://github.com/Entrpi/sparkinfer-glmrt)
`glm-next-backport` @ `2f53ce3`; exllamav3 `c5d9c657`; FlashInfer 0.6.18
`083012d6` + 2 sm12x patches (shipped). No unlisted patches.

**Changes from base:** README "How it works" and FINDINGS §18–19. This
branch adds to the local-inference-lab fork lineage: the GLM-5.3-Flash
port with sm121 fixes, fused serving of the standard EXL3 checkpoint,
the DFlash2 speculative lane (salted-Gumbel draft sampling, non-causal
draft attention, mHC aux capture, draft-KV grouping and a ring-backed
draft KV), chat-template repairs (multimodal and `enable_thinking`
gating), the FlashInfer fp8-MLA sm12x patches, fine-grained prefix reuse,
the mamba spec-state ring, the fair mixed-prefill scheduler,
boundary-state retention, the right-sized sparse-indexer workspace, and —
new in v2.3 — the drafter-step work from a torch-profiler kernel census:
eight fixed NCCL channels, the b12x MXFP8 GEMM for the drafter with a
per-row-count FlashInfer fallback, a rowwise-fp8 draft head, and the W4A16
route-pack fast path (single-stream decode step 115.5 → 110 ms at
unchanged acceptance). Inherited: b12x runtime, breakable CUDA graphs,
kpool sparse attention.

**Tested configuration:** 2× GB10 (DGX Spark), TP2 over 200GbE RoCE,
brandonmusic EXL3/TR3 4bpw (120 shards), `--quantization exl3`, DFlash2
`num_speculative_tokens=7` (MTP-4 fallback; the installer now fetches
local-inference-lab's MXFP8 copy of the IncoAI drafter — acceptance parity
with the bf16 original, FINDINGS §19), CUDA graphs on,
`gpu-memory-utilization 0.85`, `kv-cache-memory 14.4e9`, MNBT 8192, NFS
and local-weights topologies. Shipping default: `--max-model-len 524288`,
`fp8_ds_mla` packed KV + `B12X_MLA_SPARSE` (sliding-window layers
excepted), `--max-num-seqs 4`, instanttensor loader (~3.6 min
launch-to-API), fine-grained prefix hashing at 512, spec-state ring on,
`NCCL_MIN/MAX_NCHANNELS=8`, `VLLM_USE_B12X_FP8_GEMM=1`
(`VLLM_B12X_MXFP8_MAX_M=16`), `VLLM_DFLASH_FP8_DRAFT_HEAD=1` —
1,287,194-token pool (2.46 × 524k banks). Opt-in profiles: NVFP4 KV
(+28.6% pool), native 1M (2.05 concurrent full banks) per FINDINGS §13,
and the agentic profile (`MAX_SEQS` 12–16, `SPEC=none`,
`MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512`) per §17.

**Validation results:** 20/20 self-containment checks
(`scripts/check_image.sh`) and 87/87 unit tests in the pulled image
(`scripts/run_tests.sh`, incl. the per-M dispatch and fp8-head top-k
parity tests); on the shipping default: math_500 91% (n=100) and
46 of 50 (92%) on the shipping v2.3 boot with the MXFP8 drafter, gpqa_diamond 70% (n=50), estonia
133k retrieval 10/10, lavd ledger audit 15/30 EXACT; speculative
equivalence lossless-up-to-ties (`tools/dflash_equiv.py`); same-content
concurrency arm on the v2.3 boot (16k contexts, 60 s windows, `MAX_SEQS=10`):
c1 21.6 tok/s at 9.12 steps/s (accept 2.36), c5 60.7 tok/s, c10 77.0
tok/s, prompt reading 1,494–1,598 tok/s aggregate; single-request
workloads on the shipping default with the MXFP8 drafter
(`scripts/bench_workloads.py`, medians of five 400-token runs): prose
30.4, code 41.6, JSON 50.7, math 44.9, structured 70.9 tok/s at
108–111 ms per step, TTFT 0.30–0.41 s. v2.3 receipts:
single-stream step 115.5 → 110 ms at accept parity, four streams neutral
with acceptance back in the FlashInfer band (FINDINGS §19).

**Known limitations:** on pre-v2.3 images set `VLLM_USE_B12X_FP8_GEMM=0`
(they lack the per-M fallback and lose ~0.19 accepted tokens per draft at
4 streams with it on); `--prefix-match-unit` below 4608 is ENGINE-FATAL on
`v2-glmnext` and older; the 14.4 GB KV default assumes v2.2+'s right-sized
indexer workspace — pin `KV_CACHE_MEMORY=12.4e9` on older images; the
spec-state ring's fixed cost scales with `--max-num-seqs` and is charged
inside the KV budget (16 streams at 131k need >= 16.5e9; the
`_check_enough_kv_cache_memory` boot error is the tell); the fp8 draft head
costs 317 MB per rank; the head box serves with 1–3 GB of host memory to
spare — do not co-locate other workloads; video inputs untested; contexts
past 524,288 need the NVFP4 profile; drafterless runs on v1 images should
cap `--max-model-len` at ~358k; interior 512-unit states between block
boundaries are by design not retained, and warm turns pay a fixed
2,048-token replay reserve to keep the drafter active.

**Support contact or issue tracker:**
[github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues)
(maintainer: Entrpi). Debug against this derivative before assigning
problems upstream.

**Upstreaming:** focused PRs against the local-inference-lab fork are
planned, strongest first: the mixed-prefill share gate + chunk cap, the
mamba eagle-backoff fix + fine-restore draft guard, boundary-state
retention, the indexer workspace call-site asymmetry (with the deepseek_v4
precedent and MiaAI-Lab credit), the spec-state ring, FlashKDA-for-GLM,
eagle draft-group annotation, ring draft-KV + glm5 draft grouping, the
per-M MXFP8 dispatch and fp8 draft head, the sparkinfer b12x fixes
(route-pack fast path) to their lineage repos, and the sm121
`persistent_topk` retry.
