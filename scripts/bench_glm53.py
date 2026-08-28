#!/usr/bin/env python3
"""Benchmark GLM-5.3-Flash per forum-thread conventions (jack6464 post 22):
~161 prompt tokens, 256 forced output tokens, temperature 0.
Reports c1 decode tok/s + TTFT, and cN aggregate output tok/s. Stdlib only."""
import json, os, sys, time, threading, urllib.request

URL = "http://localhost:8000/v1/chat/completions"

# ~160-token prompt (measured against the served tokenizer via usage on first run)
PROMPT = os.environ.get("BENCH_PROMPT") or (
    "Summarize the following situation and then list three concrete recommendations. "
    "A small robotics startup has deployed a fleet of forty autonomous delivery robots "
    "across three university campuses. The robots navigate sidewalks using a combination "
    "of lidar, stereo cameras, and pre-mapped routes. During the winter months the team "
    "observed a sharp increase in navigation failures: wheel slip on icy inclines, lidar "
    "returns degraded by falling snow, and camera exposure problems caused by low sun "
    "angles. Support tickets tripled, and two robots were temporarily lost after their "
    "localization drifted more than fifty meters. The operations team currently monitors "
    "the fleet through a dashboard that refreshes every thirty seconds and alerts only on "
    "battery faults. Management wants a plan before next winter."
)

def one_request(results, idx):
    body = json.dumps({
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256, "temperature": 0, "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                if ch.get("delta", {}).get("content") or ch.get("delta", {}).get("reasoning_content"):
                    if t_first is None:
                        t_first = time.time()
                    n += 1
    t_end = time.time()
    out_toks = usage["completion_tokens"] if usage else n
    results[idx] = {
        "ttft": (t_first - t0) if t_first else None,
        "decode_tps": (out_toks - 1) / (t_end - t_first) if t_first and t_end > t_first else None,
        "out_toks": out_toks, "wall": t_end - t0,
        "prompt_toks": usage.get("prompt_tokens") if usage else None,
    }

def run_c(n):
    results = [None] * n
    t0 = time.time()
    threads = [threading.Thread(target=one_request, args=(results, i)) for i in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    agg = sum(r["out_toks"] for r in results if r) / wall
    return results, agg, wall

def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None

if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    # warmup
    r = [None]; one_request(r, 0)
    print(f"warmup: prompt_toks={r[0]['prompt_toks']} out={r[0]['out_toks']} "
          f"ttft={r[0]['ttft']:.3f}s decode={r[0]['decode_tps']:.2f} tok/s", flush=True)
    c1_tps, c1_ttft = [], []
    for i in range(runs):
        res, agg, wall = run_c(1)
        c1_tps.append(res[0]["decode_tps"]); c1_ttft.append(res[0]["ttft"])
        print(f"c1 run{i+1}: decode={res[0]['decode_tps']:.2f} tok/s ttft={res[0]['ttft']:.3f}s", flush=True)
    c5_aggs = []
    for i in range(runs):
        res, agg, wall = run_c(5)
        per = [f"{r['decode_tps']:.1f}" for r in res if r]
        c5_aggs.append(agg)
        print(f"c5 run{i+1}: aggregate={agg:.2f} tok/s wall={wall:.1f}s per-req=[{','.join(per)}]", flush=True)
    print(f"\nMEDIANS: c1 decode={median(c1_tps):.2f} tok/s | c1 TTFT={median(c1_ttft):.3f}s "
          f"| c5 aggregate={median(c5_aggs):.2f} tok/s", flush=True)
