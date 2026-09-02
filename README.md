# GLM-5.3-Flash EXL3 on two DGX Sparks

Run [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), a
320-billion-parameter open model, on a pair of NVIDIA DGX Spark desktops as
a private, OpenAI-compatible API. One installer and one large download, and
you have a model of frontier-class scale answering on your own hardware with
1.3 million tokens of context memory at full quality (1.7 million with the
compact-memory option), vision, tool calling, and quality at parity with the
official release.

**Status: community derivative.** Not an official image of any upstream
project. Built and maintained by [Entrpi](https://github.com/Entrpi); report
problems on [this repo's issue tracker](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/issues)
before assigning them upstream.

## What you get

| | |
|---|---|
| Model | GLM-5.3-Flash, 320B parameters (18B active per token), served from a 4-bit EXL3 checkpoint independently measured at parity with the official FP8 release |
| Context memory | 1,287,194 tokens of attention memory at parity quality, shared across whatever is in flight; 1,702,584 with the compact-memory option (small quality step); 2,144,814 in the 1M-request mode. The per-request limit is a flag (`MAX_LEN`, default 524,288: about 2.5 such requests at once, or many shorter ones) |
| Speed | about 32 tokens/s for a single chat, 70+ tokens/s on structured output (JSON, lists, code), first token in about 0.4 s; a full 500k-token document is read in about 6.5 minutes |
| Quality | math_500 91%, GPQA-diamond 70%, exact retrieval from a 133k-token document 10/10, all measured on this exact setup |
| Modalities | text, images, tool calling, optional visible reasoning |
| API | OpenAI-compatible (`/v1/chat/completions`, `/v1/models`), works with any OpenAI client |
| Hardware | 2× DGX Spark (GB10, 128 GiB each) joined by their 200GbE ports |

Every number above is measured, not estimated; the tables further down say
how.

## Quick start

You need two DGX Sparks connected directly by their QSFP ports, Docker with
GPU support on both, and passwordless SSH from the box that will serve the
API (the *head*) to the other one (the *worker*).

On the head:

```bash
git clone https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark
cd glm-5.3-flash-exl3-2x-spark
cp .env.example .env    # set the worker's LAN address and the two rail IPs/interfaces
./install.sh
```

The installer checks both machines, pulls the ~25 GiB serving image, downloads
the ~176 GiB model and ~1.3 GiB drafter, installs the launch scripts, starts
the worker and then the head, warms up, and runs a test request. Every step
is safe to re-run. Downloads dominate the first run; after that a restart
takes about four minutes to a live API.

```bash
./install.sh --nfs             # keep the weights only on the worker (head needs ~30 GiB disk)
./install.sh --skip-download   # weights already in place
./install.sh --help            # all options and exactly what the script touches
```

Then talk to it:

```bash
curl -s http://<head-ip>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "Summarize the attached contract in five bullets."}]
}'
```

Any OpenAI SDK works with `base_url="http://<head-ip>:8000/v1"` and the model
name `glm-5.3-flash`.

### Requirements

| | |
|---|---|
| Machines | 2× DGX Spark (GB10, 128 GiB unified memory each) |
| Link | direct 200GbE connection between the two QSFP ports; the two boxes talk over this link only |
| Disk | ~200 GiB free per box (or ~200 GiB on the worker and ~30 GiB on the head with `--nfs`) |
| Software | Docker with the NVIDIA runtime on both; passwordless SSH head → worker |
| Swap | not needed with the default loader; if you switch to the page-cached loader, grow swap to 32 GiB on both boxes first (the installer offers to) |
| Free memory | under 6 GB of system memory in use on each box before launch (a headless box with no desktop session); the installer and the launcher both check and refuse otherwise, naming the consumers. `MEM_USED_MAX_GB=<gb>` raises the limit, `0` disables the check (`--force` on the installer downgrades it to a warning), and a smaller `KV_CACHE_MEMORY` is the safe way to serve on a busier box |

## Using the model

**Reasoning is off by default** so answers arrive as plain content. Turn it
on per request and the reasoning comes back in `message.reasoning`, never
mixed into the answer:

```json
{"model": "glm-5.3-flash", "messages": [...], "chat_template_kwargs": {"enable_thinking": true}}
```

**Tool calling** follows the OpenAI `tools` / `tool_choice` schema and works
with reasoning on or off. **Images** go in as standard `image_url` content
parts. Greedy decoding (`temperature: 0`) is fully supported and is what the
quality numbers below use.

**Long documents.** A 133k-token prompt is processed in about 90 s, a
500k-token one in about 6.5 minutes. Repeated turns on the same document
reuse the work already done: a follow-up question on a 15k-token
conversation answers in about 3 s instead of 12.

**Capacity.** What limits you is the attention-memory pool, 1,287,194
tokens at the default quality, shared by everything in flight; the
per-request maximum (`MAX_LEN`, 524,288 by default) is just a cap on how
much of it one request may take. The default configuration admits up to
four requests at once and aggregate throughput rises with concurrency
(about 47 tokens/s across four chats). For many short concurrent sessions,
or for 1M-token requests, see the configurations below.

## Choosing a configuration

The defaults are the validated production setup. Everything else is an
environment variable set in `.env` (or on the launch command line) on both
boxes. Pick by what you serve:

| You mostly want | Set | You get | You give up |
|---|---|---|---|
| **Long documents, a few users** (default) | nothing | 1,287,194-token pool at full quality; requests up to 524k (2.5 of those at once, or more shorter ones); the fastest single-stream decode | — |
| **Snappy chat while others upload documents** | `MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512` | a running chat keeps 85–91% of its speed while a cold 100k-token upload is processed, with pauses of ~0.6 s instead of ~3 s | document processing slows by 20–35% while chats are active; no cost when idle |
| **More context memory** | `KV_DTYPE=nvfp4_ds_mla VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` | 1,702,584-token pool (+32%): 3.25 requests at 524k | a small quality step (math 88 vs 91) |
| **1M-token requests** | the row above plus `MAX_LEN=1048576 MNBT=4096` | 2,144,814-token pool; requests up to 1,048,576 tokens, two of them at once; exact retrieval verified at 875k depth | ~15 minutes to read a full 1M-token prompt |
| **Many concurrent short sessions** (agents, batch jobs) | `MAX_LEN=131072 MAX_SEQS=12 MNBT=4096 SPEC=none MIXED_PREFILL_DECODE_WEIGHT=1.0 MIXED_PREFILL_CAP=512` | 12–16 streams at 88–103 tokens/s aggregate; without the drafter the pool grows to 1,858,451 tokens at the same budget (+44%) and its 1.3 GB of weights stay unloaded | per-stream speed; requests capped at 131k |
| **Unquantized attention memory** | `MAX_LEN=131072 KV_DTYPE= ATTN_BACKEND= SKIP_MM_PROFILING=0 MAX_SEQS=6` | 520,470-token pool with no 8-bit attention memory; requests up to 131k | the long banks; quality is the same within noise |
| **Fallback if the speculative drafter misbehaves** | `MTP=4` | the model's built-in multi-token head; the 1.3 GB drafter and its draft attention memory are not loaded, so the pool grows toward the no-speculation figure above (MTP-4's exact pool is not re-measured on this release) | about 20% slower decode |

Knobs used above: `MAX_LEN` is the longest request accepted; `MAX_SEQS`
how many requests run at once; `MNBT` how many prompt tokens are processed
per step while reading a document (smaller keeps chats smoother, larger
reads faster); `SPEC=none` turns the speculative drafter off; `KV_DTYPE`
picks the attention-memory format; the two `MIXED_PREFILL_*` knobs control
how document reading and live chats share the GPU.

The memory budget for context (`KV_CACHE_MEMORY`, 14.4 GB) and the GPU
memory fraction (`GMU`, 0.85) were validated for stability across the
configurations above on lightly loaded headless Sparks that had under 6 GB
of system memory in use before the server started; the head then serves
with about 1–3 GB to spare. DGX Spark unified memory does not fail
gracefully when oversubscribed, so a box with more resident services, a
desktop session, or a different firmware state may need a smaller budget:
lower `KV_CACHE_MEMORY` (each 1 GB is about 90k pool tokens) and check
headroom with [tools/memlog.sh](tools/memlog.sh) before raising anything.

## Operating it

| Task | How |
|---|---|
| Health | `curl -s http://<head-ip>:8000/health` → 200 |
| Logs | `docker logs -f vllm_glm53` on either box |
| Restart | on the head `docker rm -f vllm_glm53`, then on the worker `~/launch-glm53-vllm-tp2.sh 1`, then on the head `~/launch-glm53-vllm-tp2.sh 0` (always head down first, worker up first; ~4 min to a live API) |
| Warm-up | `~/glm53-warmup.sh` after a restart compiles the hot shapes so the first real request does not pay ~7 s |
| Update | `./install.sh` again pulls the current image; to pin a specific one, set `IMAGE=ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:<tag>` on both launches |
| Roll back | `IMAGE=ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.2-ring VLLM_USE_B12X_FP8_GEMM=0 VLLM_DFLASH_FP8_DRAFT_HEAD=0` on both launches |
| Per-box config | `~/.glm53-serve.env` (written from `.env` on every install run, so keep your changes in `.env`); command-line variables always win |
| Upgrade from an earlier release | `git pull`; in `.env` remove any `IMAGE=` line and any `GLM53_TOPK_FIX_SO`; move aside any `~/glm53-hotfix*` directories on both boxes (the installer stops if they exist); then `./install.sh --skip-download`, optionally with `--prune-old-images` to reclaim ~18 GiB per superseded release on each box. Pins to older images are refused with the fix printed |
| Verify an image | `bash scripts/check_image.sh` (self-containment) and `bash scripts/run_tests.sh` (87 unit tests inside the image; run while the GPUs are free) |

## Measured performance

Single request, greedy decoding, on the default configuration. Each figure
is the median of five 400-token runs; the range shows the run-to-run spread
(the drafter accepts more or fewer tokens depending on the exact wording the
model chooses, so two runs of the same prompt differ by a few percent):

| Workload | Tokens/s | Run-to-run range |
|---|---:|---:|
| Prose, Q&A | 30 | 29–32 |
| Code | 42 | 41–43 |
| JSON | 51 | 50–52 |
| Math | 45 | 41–50 |
| Structured output (lists, counting) | 71 | 70–72 |
| First token (short prompt) | 0.3–0.4 s | |
| Reading a document | 1,490 tokens/s at 133k; 1,277 tokens/s over a full 499k bank | |

Speed depends on how predictable the text is: the model runs a small
drafter that proposes up to seven tokens per step and keeps the ones the
full model agrees with, so repetitive or structured output runs two to
three times faster than free prose. Temperature and reasoning mode do not
change this.

Several requests at once, each with its own 16k-token prompt, all submitted
at the same moment (low-predictability synthetic text, so per-request decode
sits below the prose figure above). Time to first token here includes
waiting for the other prompts to be read; a short prompt on an idle server
answers in about 0.4 s. Measured on this release with `MAX_SEQS=10` so all
ten could be admitted:

| Concurrent requests | Decode, aggregate tokens/s | Decode per request | Time to first token, median / last | Prompt reading, aggregate tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 21.6 | 21.6 | 9.9 s / 9.9 s | 1,598 |
| 5 | 60.7 | 10.2 | 38.7 s / 53.2 s | 1,494 |
| 10 | 77.0 | 7.6 | 65.1 s / 101.2 s | 1,575 |

The agentic configuration (no drafter, 12–16 streams) reaches 88–103
tokens/s aggregate at 8k-token contexts.

Context capacity by configuration (tokens of attention memory available,
and how many maximum-length requests fit at once):

| Configuration | Pool (tokens) | Full-length requests at once |
|---|---:|---:|
| Default (524k) | 1,287,194 | 2.46 |
| NVFP4 attention memory (524k) | 1,702,584 | 3.25 |
| Native 1M | 2,144,814 | 2.05 |
| 131k, unquantized attention memory | 520,470 | 3.97 |

## Measured quality

Greedy decoding, local harness with robust answer extraction, on the default
configuration:

| Test | Result |
|---|---|
| math_500 (n=100) | 91% |
| GPQA-diamond (n=50) | 70% |
| Retrieval from a 133,186-token document (n=10) | 10/10 |
| Ledger audit over a long document (n=30, exact / near / fail) | 15 / 14 / 1 |
| Speculative decoding equivalence | lossless up to argmax ties ([tools/dflash_equiv.py](tools/dflash_equiv.py)) |
| Tool calls and image inputs | verified end to end |

An independent KL-divergence panel puts the 4-bit weights at parity with
the official FP8 release; the math and retrieval gates above are at parity
with an unquantized-attention-memory run of the same model on this
hardware.

## How it compares

[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
serves the same model on the same hardware through a different engine.
On their own benchmark protocol (vendored unmodified as
[scripts/bench_decode_miaai.py](scripts/bench_decode_miaai.py)):

| | This recipe | MiaAI (published) |
|---|---:|---:|
| Structured output, tokens/s | 74.7 (74.1–76.5) | 61.7–62.9 |
| Prose, tokens/s | 31.1 (30.7–32.0) | 26.9 |
| First token | 0.43–0.46 s | ~0.72 s |
| Attention memory at the same precision | +46% | — |

Both projects arrived at the same three essential fixes independently; the
differences are in engine lineage, memory format, and how much context is
kept available per request. Full analysis in
[docs/COMPARISON.md](docs/COMPARISON.md).

## How it works

The two Sparks split the model between them (tensor parallel over the
200GbE link). The 288-expert mixture-of-experts weights are served directly
from a 4-bit trellis-quantized checkpoint, with only the parts a token
actually routes to read from memory. A one-billion-parameter drafter
([DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), installed as
the [8-bit copy](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8)
published by local-inference-lab, which accepts exactly as many tokens and
runs a few percent faster) proposes
blocks of tokens that the full model verifies in one pass. Attention memory
is stored as compact 8-bit records so that half a million tokens of context
fit, and the whole decode step is captured as CUDA graphs. The engine is a
vLLM fork branch, [Entrpi/vllm-glm-5.3-flash-spark](https://github.com/Entrpi/vllm-glm-5.3-flash-spark),
baked into the image `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark`.

The engineering record, including what did not work, is
[docs/FINDINGS.md](docs/FINDINGS.md); the image's exact provenance (base,
commits, build command, patches) is [docs/BUILD.md](docs/BUILD.md).

## Repository layout

```
install.sh                      one-shot installer (run on the head)
.env.example                    topology and knobs; copied to .env on first run
scripts/
  launch-glm53-vllm-tp2.sh      per-box launcher (0 = head, 1 = worker)
  glm53-warmup.sh               post-boot warm-up
  bench_glm53.py                single-stream / 5-stream / first-token benchmark
  bench_workloads.py            decode speed by workload: median, spread, acceptance
  bench_decode_miaai.py         MiaAI's protocol, vendored for comparability
  sweep_runner.py               concurrency sweep with identical content per arm
  saturation_bench.py           scheduler-pressure benchmark
  longctx_smoke.py              long-prompt processing + drafting probe
  warm_prefix_test.py           repeated-turn reuse test
  accept_ab_matrix.py           temperature × reasoning acceptance A/B
  check_image.sh                image self-containment verification
  run_tests.sh                  unit tests inside the pulled image
tools/
  dflash_equiv.py               speculative-decoding equivalence harness
  memlog.sh                     memory-headroom sampler
docs/
  FINDINGS.md                   engineering record, measurements, negative results
  COMPARISON.md                 cross-stack comparison with the MiaAI recipe
  BUILD.md                      image provenance: commits, build args, patches
  ANNOUNCEMENT.md               community publishing disclosure
```

## Reproducing the numbers

```bash
python3 scripts/bench_glm53.py 3                                  # single/5-stream/TTFT, 3-run medians
python3 scripts/bench_workloads.py 5 400                          # per-workload table: median, spread, acceptance
python3 scripts/bench_decode_miaai.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence
python3 scripts/bench_decode_miaai.py --phase prose --runs 5 --max-tokens 400 --skip-coherence
python3 scripts/sweep_runner.py --arm mine --cells 1,2,4          # concurrency table
python3 scripts/longctx_smoke.py                                  # long-prompt processing
python3 tools/dflash_equiv.py                                     # speculative equivalence
bash scripts/run_tests.sh && bash scripts/check_image.sh          # GPUs must be free
```

Quality gates (math_500, GPQA-diamond) used a local harness with greedy
decoding and robust answer extraction, n = 50–100 per gate; methodology in
[docs/FINDINGS.md](docs/FINDINGS.md).

## Related work

| Project | Role |
|---|---|
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | parallel recipe on a different engine; benchmark protocol vendored here |
| [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) | the vLLM fork lineage this engine builds on |
| [tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt) | the b12x GPU runtime the fused expert kernels come from |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | the EXL3 4-bit format and kernels |
| [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | the 4-bit checkpoint served here |
| [local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8) | the drafter as installed: an 8-bit copy of IncoAI's DFlash2 (CC BY-NC-ND 4.0; downloaded from its source, never redistributed here) |
| [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | the original 16-bit drafter (same license); `DFLASH_REPO=incoai/GLM-5.3-Flash-DFlash2 DFLASH_DIR=$HOME/models/glm53-dflash2` installs it instead |
| [Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark) | sibling single-Spark recipe |
| [Entrpi/dgx-spark-serving-mode](https://github.com/Entrpi/dgx-spark-serving-mode) | reclaim desktop-held unified memory before serving |

## Acknowledgements

- **brandonmusic** — the 4-bit quant this recipe serves
- **IncoAI** — the DFlash2 drafter
- **local-inference-lab** — the 8-bit (MXFP8) drafter copy and the NVFP4 model conversions
- **eugr** — the `spark-vllm-docker` build system the image is built with
- **Mia's AI Lab** — the parallel recipe, the benchmark protocol, the checkpoint mirror, and several operational ideas carried over from it
- **tonyd2wild, jack6464, malaiwah** — forum groundwork and the quality panel
- **local-inference-lab, tpurtell, turboderp** — the runtimes underneath

## License

This repository (installer, scripts, docs) is MIT. The vLLM branch inherits
Apache-2.0. Model and drafter weights keep their upstream licenses; the
DFlash2 drafter is **CC BY-NC-ND 4.0** (research/eval), so the installer
downloads it from its source repository and this project never redistributes
it.
