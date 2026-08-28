"""Saturation bench: 6 concurrent ~8k-token-prompt requests, 512-token
outputs, on the DFlash2 server — scheduler behavior when the reduced 329k
pool is under real pressure. Reports per-request wall/tok-s, aggregate,
batch accept delta; check the server log afterwards for preemptions.
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import os
API = os.environ.get("API_BASE", "http://localhost:8000")
MODEL = "glm-5.3-flash"

para = (
    "The relay station processes telemetry frames in fixed windows, applying "
    "vector quantization to each channel before forwarding summaries to the "
    "archive tier, where retention policies rotate cold segments weekly. "
)
DOC = para * 260  # ~8k tokens


def scrape():
    acc = dr = 0.0
    txt = urllib.request.urlopen(API + "/metrics", timeout=10).read().decode()
    for ln in txt.splitlines():
        if ln.startswith("#"):
            continue
        if "spec_decode_num_accepted_tokens_total" in ln:
            acc += float(ln.split()[-1])
        elif "spec_decode_num_drafts_total" in ln:
            dr += float(ln.split()[-1])
    return acc, dr


def one(i):
    prompt = (f"Document copy {i}:\n{DOC}\n\nTask {i}: Summarize the "
              f"document's pipeline in about 400 words, then list 5 risks.")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 512, "seed": i,
    }).encode()
    req = urllib.request.Request(
        API + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    el = time.perf_counter() - t0
    u = r["usage"]
    return i, u["prompt_tokens"], u["completion_tokens"], el


a0, d0 = scrape()
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(one, range(6)))
wall = time.perf_counter() - t0
a1, d1 = scrape()

total_out = sum(ct for _, _, ct, _ in results)
for i, pt, ct, el in results:
    print(f"  req{i}: prompt={pt} out={ct} wall={el:.1f}s = {ct / el:.1f} tok/s")
drafts = d1 - d0
alen = 1 + (a1 - a0) / drafts if drafts > 0 else 0.0
print(f"aggregate: {total_out} tokens in {wall:.1f}s = {total_out / wall:.1f} "
      f"tok/s | batch accept {alen:.2f}/8 ({drafts:.0f} drafts)")
