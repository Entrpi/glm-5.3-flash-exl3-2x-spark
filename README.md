# GLM-5.3-Flash EXL3 + DFlash2 on 2× DGX Spark

OpenAI-compatible vLLM serving of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(320B/A18B) on two NVIDIA GB10 DGX Sparks over a 200GbE rail — 4-bit EXL3
weights, DFlash2 block-diffusion speculative decode, CUDA graphs, fused
trellis MoE, vision and tool calling included, quality-gated at every step.

**Status: community derivative.** This is not an official image of any
upstream project. Built and maintained by [Entrpi](https://github.com/Entrpi);
report problems on [this repo's issue tracker](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues)
(please debug against this derivative before assigning a problem upstream).

**Headline numbers** (temperature 0 unless noted): ~33–35 tok/s single-stream
prose decode (+21% over the MTP-4 head), **72 tok/s** on high-acceptance
structured output, TTFT 0.42–0.51 s, 112k-token prefill in ~55 s,
math_500 **94%** (n=50), gpqa_diamond **78%**, vision verified end-to-end
including tool calls. Every claim is cashed out in a table below.

- **Model**: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — 320B MoE, 18B active, hybrid KDA + sparse-MLA attention
- **Quant**: [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) (EXL3/TR3 uniform-K4, ~176 GiB; independent KLD panel puts it at parity with the official FP8 release)
- **Drafter**: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (1B block-diffusion draft, CC BY-NC-ND 4.0 — always downloaded from its source repo, never redistributed here)
- **Engine**: [Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark) (vLLM fork branch) baked into `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark` — provenance in [docs/BUILD.md](docs/BUILD.md)
- **Hardware**: 2× NVIDIA GB10 (DGX Spark, 128 GiB unified each), direct 200GbE QSFP link

## Quick start

On the head box (the one that will serve the API), with passwordless SSH to
the worker:

```bash
git clone https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark
cd glm-5.3-flash-exl3-2x-spark
cp .env.example .env    # set WORKER_LAN_IP + the rail IPs/interfaces
./install.sh
```

This command: (1) verifies both hosts, (2) pulls the ~25 GiB serving image
and ships it to the worker, (3) downloads the ~176 GiB EXL3 weights and the
~2.3 GiB drafter, (4) installs the launch scripts and per-box config,
(5) launches worker then head (~13 min to API), warms the JIT shapes, and
smoke-tests the endpoint. Every step is idempotent — re-run freely.

Variants:

```bash
./install.sh --nfs             # weights only on the worker, NFS to the head
```

```bash
./install.sh --skip-download   # weights already in place
```

```bash
./install.sh --help            # all flags + what the script touches
```

## Hardware requirements

| | Minimum |
|---|---|
| Boxes | 2× DGX Spark (GB10, 128 GiB unified memory each) |
| Interconnect | direct 200GbE QSFP link (RoCE); NCCL runs on this rail |
| Disk, default (local) mode | ~200 GiB free per box |
| Disk, `--nfs` mode | ~200 GiB worker, ~30 GiB head |
| Software | Docker with GPU support on both; passwordless SSH head→worker |

## Memory and context

Serving defaults to a **524,288-token context** with fp8 KV (since
2026-08-29). Measured pools on this stack, all at the same pinned 12.4 GB
KV budget:

| Configuration | KV pool (tokens) | Concurrency @max-len | Notes |
|---|---:|---:|---|
| **Default: 524k, DFlash2 + `fp8_e4m3` KV** | **1,435,070** | 2.74× | long-context production config; vLLM auto-bumps the KV block to 4608 for KDA/attention page parity |
| 358k, DFlash2 + fp8 | 1,275,306 | 3.56× | `MAX_LEN=358400` — more concurrency headroom |
| 131k short-context mode, bf16 KV | 520,470 | 3.97× | `MAX_LEN=131072 KV_DTYPE= SKIP_MM_PROFILING=0 MAX_SEQS=6` — best math quality |
| 131k, fp8 (2304 block) | 769,817 | 5.87× | historical option gate |
| 131k, no speculation, bf16 | 917,504 | 7.00× | `SPEC=none` |

**The fp8 trade, measured:** long-context quality is at parity with bf16
(estonia 133k-token retrieval 9/10 — community band; lavd ledger-audit
n=30 statistically identical; 111k needle all-facts-exact), and math_500
shows no regression against current-template bf16 (fp8@524k: 87–88%;
same-day bf16 re-measure: 80% n=50 — the older 94% figure predates the
`enable_thinking` template fix and is not comparable). bf16 cannot reach
the long banks — its pool is ~0.99× a single 524k request. Estonia prefill: 133,186 tokens at
~1,400 tok/s through the default 8192-token chunks.

The 12.4 GB KV budget is pinned deliberately. Explicit budgets bypass vLLM's
memory profiling reserve, and GB10 unified memory does not fail gracefully —
it swap-wedges. The pinned value keeps a **measured 5.25 GiB minimum free**
on the memory-binding box under saturation plus a 112k-token prefill.

> Raising the budget is non-linear: 13.4 GB bought +47k tokens but collapsed
> the measured floor from 5.25 GiB to 2.26 GiB. Do not raise
> `KV_CACHE_MEMORY` or `GMU` without re-running the floor methodology
> ([tools/memlog.sh](tools/memlog.sh)) — floors also degrade 1.5–2 GiB per
> day of workload, so fresh-boot numbers are optimistic.

> In default (local-weights) mode the head loads the checkpoint from local
> NVMe. On our reference kit, a head-side fast local read outran unified-
> memory page-cache reclaim and wedged the box at ~90% of shard load — that
> box now runs `--nfs` mode (weights on the worker, NFS-paced load), which
> is the topology all production numbers here were measured on. If your head
> wedges during load, re-run `./install.sh --nfs`.

### Startup time

Worker joins in ~25 s; the head takes **~12–13 min** to API-up (NFS-paced
weight load + engine init + CUDA-graph capture, including the drafter's own
full graphs). `install.sh` then runs a ~20 s JIT shape warmup so the first
real request doesn't pay ~7 s of lazy compilation. JIT caches persist across
relaunches (`~/glm53-vllm-cache/jit`).

## Profiles

| Profile | How | When |
|---|---|---|
| **Default** | (nothing) | 524k context, DFlash2 k=7 + fp8 KV + CUDA graphs — 2.74 concurrent full banks |
| Short-context bf16 | `MAX_LEN=131072 KV_DTYPE= SKIP_MM_PROFILING=0 MAX_SEQS=6` on both launches | the pre-2026-08-29 production config; 131k context |
| MTP fallback | `MTP=4` on both launches | if the drafter ever misbehaves; ~21% slower |
| No speculation | `SPEC=none` | maximum KV pool, debugging |

Thinking is **off by default** at the serving layer
(`--default-chat-template-kwargs '{"enable_thinking": false}'`); enable per
request with `"chat_template_kwargs": {"enable_thinking": true}` — reasoning
then arrives in `message.reasoning`, never mixed into `content`. Tool
calling (`glm47` parser) and vision (image inputs) work in both modes.

## Benchmarks

### DFlash2 vs the native MTP head (this stack, single variable)

| Config | c1 prose tok/s | c5 aggregate | TTFT | math_500 n=50 |
|---|---:|---:|---:|---:|
| EXL3 fused + graphs, MTP-4 | 28.6 | 91.7 | 0.422 s | 90% |
| **EXL3 fused + graphs, DFlash2 k=7** | **33.1–34.7** | 83.7–96.0 | 0.50 s | **94%** |

DFlash2 accepts 3.9–5.5 of 8 positions on mixed work versus MTP-4's 3.8 of
5 — the deeper block wins ~21% single-stream. MTP-5 was also tested and
regresses (prose acceptance collapses at position 5); MTP-4 remains the best
fallback. Ranges are boot-to-boot variance across three validated boots.

### Acceptance by workload (DFlash2, k=7, accept length of 8 possible)

| Workload | Accept | e2e tok/s |
|---|---:|---:|
| Prose / Q&A | 3.5–3.6 | 30.5 |
| Code | 4.1 | 34.9 |
| JSON | 4.7 | 40.3 |
| Math | 5.8 (6.8–7.1 @ temp 1) | 37.7 |
| Long-context code | 5.2 | 43.8 |
| Doc-grounded QA / multi-turn | 5.3–7.0 | — |
| Structured counting | 7.8 (0.98/pos) | 72.4 |

Acceptance — not kernel speed — is the whole game: the engine steps at
~115 ms regardless, so tok/s ≈ (1 + accepted) × step rate. Prose ~3.5/8 is
the drafter's intrinsic floor; temperature and thinking mode are **not**
acceptance levers (measured: a 2×2 A/B moved the mean by <0.2).

### Cross-stack: the MiaAI-Lab recipe, on its own benchmark

[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
serves the same checkpoint + drafter on the same hardware class via a
different stack (FlashInfer sparse-MLA SM120, packed fp8 KV, 900k context).
Numbers below use **their** `bench_decode.py` protocol, vendored unmodified
as [scripts/bench_decode_miaai.py](scripts/bench_decode_miaai.py):

| Protocol phase | This stack | MiaAI (their published) |
|---|---:|---:|
| Structured (count 1→200), tok/s | **72.4** | 61.7–62.9 |
| — accept/step (of 7) | 6.86 | 6.43 |
| Prose (hash-map), tok/s | **27.4** | 26.9 |
| TTFT (short prompt) | **0.43–0.47 s** | ~0.72 s |

> A best-on-each-stack comparison, not a single-variable controlled run —
> different vLLM lineages, KV formats, and context ceilings. Their recipe
> holds a real edge this one does not attempt: 900k-token max context
> (982,612-token packed-fp8 pool). This stack counters with bf16-KV quality
> defaults, 4× the concurrency at 131k, ~8× larger prefill chunks, and the
> deeper eval story. Both projects independently converged on the same three
> hard fixes (drafter KV grouping, mHC aux capture, non-causal draft
> attention) — see [docs/COMPARISON.md](docs/COMPARISON.md).

### Quality

| Gate | Result |
|---|---|
| math_500 n=50 (this image) | **94%** (47/50) |
| math_500 n=100 (fp8-KV option gate) | 89% (bar ≥88% → fp8 allowed as option) |
| gpqa_diamond n=50 | **78%** (39/50) — best of three stacks tested on this kit |
| Speculative equivalence | lossless up to argmax tie-flips (rescoring-control methodology, [tools/dflash_equiv.py](tools/dflash_equiv.py)) |
| Vision | plain image + image-with-tool-call verified at temp 0; drafting shows **no acceptance penalty** on vision requests |

## Under the hood

The engine is a vLLM fork branch ([Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark));
everything below is baked into the image. Inherited from the
local-inference-lab fork lineage: the b12x runtime, breakable CUDA graphs,
kpool-compressed sparse attention. Introduced by this branch:

| Area | What |
|---|---|
| glm5_next port | GLM-5.3-Flash architecture (PR #53906) re-expressed in the fork's KV-tensor dialect, + sm121 day-0 fixes |
| EXL3 serving | generic tensor_storage reader + `standard_fused_moe` trellis fast path (load-time TP slicing of the standard checkpoint — no offline conversion) |
| DFlash2 | 14-file transplant of upstream's DFlash2 lane onto the fork's V1 runner, incl. salted-Gumbel draft sampling and non-causal in-block attention |
| mHC aux capture | EAGLE3-style hidden taps at layers (6,15,25,34,43) through the hyper-connection contract (`hc_post` + `hc_contract`), oracle-tested |
| Draft KV grouping | drafter's sliding-window layers get their own KV groups inside the glm5 fast path (the generic path cannot serve this hybrid) |
| Ring draft KV | fixed-size ring-backed draft KV: `1 + max_seqs × R` pages total + a positional remap kernel — recovers the pool DFlash2 otherwise costs (329k → 520k tokens) |
| Chat template | multimodal template repair for text-only-template checkpoints + `enable_thinking` gating (reasoning never leaks into `content`) |
| FlashInfer sm12x | fp8-MLA gate + `CTA_TILE_KV` cap patches (live-validated; required only for the fp8-KV option) |

41 unit tests covering the above run in the pulled image:
[scripts/run_tests.sh](scripts/run_tests.sh).

## Repo layout

```
install.sh                      one-shot installer (run on the head)
.env.example                    topology + knobs; copied to .env on first run
scripts/
  launch-glm53-vllm-tp2.sh      per-box launcher (rank 0 = head, 1 = worker)
  glm53-warmup.sh               post-boot JIT shape warmup
  bench_glm53.py                c1/c5/TTFT bench (3-run medians)
  bench_decode_miaai.py         MiaAI's protocol, vendored for comparability
  saturation_bench.py           6x ~8.6k-prompt scheduler-pressure bench
  longctx_smoke.py              112k-token prefill + accept probe
  warm_prefix_test.py           prefix-cache + drafting interaction test
  accept_ab_matrix.py           temp x thinking acceptance A/B
  check_image.sh                image self-containment verification
  run_tests.sh                  41-test suite inside the pulled image
tools/
  dflash_equiv.py               speculative-equivalence harness (rescoring)
  memlog.sh                     1 Hz memory-floor sampler
docs/
  FINDINGS.md                   the full engineering story + negative results
  COMPARISON.md                 cross-stack analysis vs the MiaAI recipe
  BUILD.md                      image provenance: commits, args, base, patches
  ANNOUNCEMENT.md               community publishing checklist disclosure
```

## Reproducing

```bash
# throughput (run on or against the head)
python3 scripts/bench_glm53.py 3
python3 scripts/bench_decode_miaai.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence
python3 scripts/bench_decode_miaai.py --phase prose --runs 5 --max-tokens 400 --skip-coherence

# acceptance + scheduler behavior
python3 scripts/saturation_bench.py
python3 scripts/longctx_smoke.py
python3 scripts/warm_prefix_test.py

# equivalence (spec-on vs spec-off, rescoring-control)
python3 tools/dflash_equiv.py

# unit tests + image verification (GPUs must be free — run before launch)
bash scripts/run_tests.sh
bash scripts/check_image.sh
```

Quality numbers (math_500 / gpqa_diamond) were produced with a local
harness: greedy decoding, robust answer extraction (never first-line
grading), n=50–100 per gate; methodology in
[docs/FINDINGS.md](docs/FINDINGS.md).

## Related work

| Project | Role |
|---|---|
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | parallel recipe, different stack; 900k context; benchmark protocol vendored here |
| [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) | the fork lineage this branch builds on |
| [tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt) | b12x trellis runtime (the fused-MoE contract this stack serves through) |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | EXL3 format + kernels |
| tonyd2wild / jack6464 (NVIDIA forums) | the original GLM-5.3-Flash-on-Spark NVFP4 recipes this project started from |
| [Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark) | sibling single-Spark recipe (DFlash1) |
| [Entrpi/dgx-spark-serving-mode](https://github.com/Entrpi/dgx-spark-serving-mode) | reclaim desktop-held unified memory before serving |

## Acknowledgements

- **brandonmusic** — the EXL3/TR3 4bpw quant this whole recipe serves
- **IncoAI** — the DFlash2 drafter
- **eugr** — the `spark-vllm-docker` build system the image is built with
- **Mia's AI Lab** — the parallel recipe, the benchmark protocol, and the
  checkpoint mirror; several operational ideas here (JIT cache persistence,
  boot warmup, thinking-off template handling) were carried over from it
- **tonyd2wild, jack6464, malaiwah** — forum groundwork and the KLD panel
- **local-inference-lab, tpurtell, turboderp** — the runtimes underneath

## License

This repository (installer, scripts, docs) is MIT. The vLLM branch inherits
Apache-2.0. Model and drafter weights keep their upstream licenses — note
the DFlash2 drafter is **CC BY-NC-ND 4.0** (research/eval): the installer
downloads it from its source repository and this project never redistributes
it.
