"""G7 concurrency-sweep cell runner (GLM-5.3-Flash 2x Spark).

Implements the ratified sweep design: per client concurrency c, start c
streams each with a UNIQUE ~16k-token context (unique from token 0 — no
warm-prefix inheritance), greedy + ignore_eos, open a fixed measurement
window once ALL streams are decoding, and report aggregate generated
tok/s plus engine-side metric deltas (steps, spec accept, preemptions).

Usage:
  glm53_sweep_runner.py --arm dflash                 # full grid
  glm53_sweep_runner.py --arm smoke --cells 1,2 --duration 20 --ctx 4000
"""

import argparse
import asyncio
import json
import random
import time

import httpx

WORDS = (
    "telemetry relay quantization archive lattice conduit manifold spectral "
    "gradient cascade polymer vertex orbital filament reactor dynamo prism "
    "catalyst turbine capacitor resonance modulation aperture spectrometer "
    "collimator interferometer oscillator waveguide attenuator amplifier "
    "substrate annealing photolithography metrology interposer chiplet"
).split()

METRIC_KEYS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:iteration_tokens_total_count",
)


def make_text(tag: str, n_chars: int) -> str:
    rng = random.Random(tag)
    parts = [f"CELL-{tag}-COLD."]
    total = len(parts[0])
    while total < n_chars:
        w = rng.choice(WORDS)
        parts.append(w)
        total += len(w) + 1
    parts.append(
        " Write a very long, detailed technical report expanding on every "
        "concept above. Do not stop."
    )
    return " ".join(parts)


async def scrape_metrics(client, base):
    out = {}
    r = await client.get(f"{base}/metrics", timeout=30)
    for line in r.text.splitlines():
        if line.startswith("#"):
            continue
        for key in METRIC_KEYS:
            if line.startswith(key + "{") or line.startswith(key + " "):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


class Stream:
    def __init__(self, idx):
        self.idx = idx
        self.tokens = 0          # chunks received (>= tokens; see note)
        self.usage_tokens = 0    # completion_tokens from streamed usage
        self.first_tok_t = None
        self.last_tok_t = None
        self.done = False
        self.error = None


async def run_stream(client, base, model, text, st: Stream, stop_evt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with client.stream(
            "POST", f"{base}/v1/chat/completions", json=payload,
            timeout=httpx.Timeout(1800.0, connect=30.0),
        ) as resp:
            async for line in resp.aiter_lines():
                if stop_evt.is_set():
                    break
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                u = d.get("usage")
                if u and u.get("completion_tokens"):
                    self_t = time.monotonic()
                    st.usage_tokens = u["completion_tokens"]
                    st.last_tok_t = self_t
                    if st.first_tok_t is None and u["completion_tokens"] > 0:
                        st.first_tok_t = self_t
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    now = time.monotonic()
                    st.tokens += 1
                    st.last_tok_t = now
                    if st.first_tok_t is None:
                        st.first_tok_t = now
    except Exception as e:  # noqa: BLE001
        st.error = f"{type(e).__name__}: {e}"
    st.done = True


async def run_cell(client, base, model, arm, c, ctx_tokens, duration, cpt,
                   max_tokens):
    streams = [Stream(i) for i in range(c)]
    stop_evt = asyncio.Event()
    texts = [
        make_text(f"{arm}-c{c}-s{i}", int(ctx_tokens * cpt)) for i in range(c)
    ]
    tasks = [
        asyncio.create_task(
            run_stream(client, base, model, texts[i], streams[i], stop_evt,
                       max_tokens)
        )
        for i in range(c)
    ]

    t_start = time.monotonic()
    while True:
        await asyncio.sleep(0.5)
        if all(s.first_tok_t is not None for s in streams):
            break
        if any(s.done and s.error for s in streams):
            stop_evt.set()
            for t in tasks:
                t.cancel()
            errs = [s.error for s in streams if s.error]
            return {"arm": arm, "c": c, "status": "stream-error", "errors": errs}
        if time.monotonic() - t_start > 1200:
            stop_evt.set()
            for t in tasks:
                t.cancel()
            return {"arm": arm, "c": c, "status": "prefill-timeout"}

    ramp_s = time.monotonic() - t_start
    m0 = await scrape_metrics(client, base)
    counts0 = [s.usage_tokens or s.tokens for s in streams]
    w0 = time.monotonic()
    deadline = w0 + duration
    while time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        newest = max((s.last_tok_t or 0) for s in streams)
        if time.monotonic() - newest > 30:
            try:
                await client.get(f"{base}/v1/models", timeout=5)
            except Exception:
                stop_evt.set()
                for t in tasks:
                    t.cancel()
                return {"arm": arm, "c": c, "status": "api-dead"}
    w1 = time.monotonic()
    counts1 = [s.usage_tokens or s.tokens for s in streams]
    m1 = await scrape_metrics(client, base)

    stop_evt.set()
    await asyncio.sleep(0.2)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    window = w1 - w0
    per_stream = [(b - a) / window for a, b in zip(counts0, counts1)]
    agg = sum(per_stream)
    md = {k: m1.get(k, 0) - m0.get(k, 0) for k in METRIC_KEYS}
    drafts = md["vllm:spec_decode_num_drafts_total"]
    accept_len = (
        1 + md["vllm:spec_decode_num_accepted_tokens_total"] / drafts
        if drafts else None
    )
    return {
        "arm": arm, "c": c, "status": "ok", "window_s": round(window, 1),
        "ramp_s": round(ramp_s, 1),
        "agg_tok_s": round(agg, 2),
        "per_stream_min": round(min(per_stream), 2),
        "per_stream_med": round(sorted(per_stream)[len(per_stream) // 2], 2),
        "engine_gen_tok_s": round(md["vllm:generation_tokens_total"] / window, 2),
        "engine_steps_s": round(
            md["vllm:iteration_tokens_total_count"] / window, 2),
        "accept_len": round(accept_len, 2) if accept_len else None,
        "preemptions": md["vllm:num_preemptions_total"],
        "prefix_hits": md["vllm:prefix_cache_hits_total"],
    }


async def probe_tokens(client, base, model, text):
    r = await client.post(
        f"{base}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": text}],
              "max_tokens": 1, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}},
        timeout=600,
    )
    return r.json()["usage"]["prompt_tokens"]


async def quality_checks(client, base, model):
    out = {}
    for name, prompt, mt in (
        ("smoke", "What is 17*23? Reply with just the number.", 16),
        ("sanity", "Explain in exactly three sentences why the sky is blue, "
                   "then list two exceptions.", 256),
    ):
        r = await client.post(
            f"{base}/v1/chat/completions",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": mt, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}},
            timeout=300,
        )
        out[name] = r.json()["choices"][0]["message"]["content"].strip()
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--host", default="192.168.88.15")
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--cells", default="1,2,4,8,12,16")
    ap.add_argument("--ctx", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    cells = [int(x) for x in args.cells.split(",")]
    limits = httpx.Limits(max_connections=40, max_keepalive_connections=40)
    async with httpx.AsyncClient(limits=limits) as client:
        cal = make_text(f"{args.arm}-cal", 40000)
        cal_tok = await probe_tokens(client, base, args.model, cal)
        cpt = len(cal) / cal_tok
        print(f"arm={args.arm} calibration {cal_tok} tok -> "
              f"{cpt:.3f} chars/tok", flush=True)

        results = []
        for c in cells:
            r = await run_cell(client, base, args.model, args.arm, c,
                               args.ctx, args.duration, cpt, args.max_tokens)
            results.append(r)
            print(json.dumps(r), flush=True)
            if r["status"] != "ok":
                print(f"ABORT: cell c={c} status={r['status']}", flush=True)
                break
            await asyncio.sleep(3)

        q = await quality_checks(client, base, args.model)
        print(f"smoke: {q['smoke']!r}", flush=True)
        out = args.out or f"/tmp/sweep_{args.arm}.json"
        json.dump({"arm": args.arm, "ctx": args.ctx,
                   "duration": args.duration, "cells": results,
                   "quality": q}, open(out, "w"), indent=1)
        print(f"saved {out}", flush=True)


asyncio.run(main())
