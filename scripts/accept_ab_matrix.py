"""Acceptance A/B matrix: temperature {0, 1} x enable_thinking {off, on}.

Same 5 albond prompts, per-request /metrics deltas on an idle server.
temp0/off is the already-measured battery baseline (rerun here anyway for a
same-session apples-to-apples row). Thinking-on rows get a larger token
budget since reasoning consumes it; acceptance is measured over whatever is
generated. temp-1 rows exercise the salted Gumbel selector walk + lossless
rejection sampling (the model card's 5.86 was measured at temp 1).
"""

import json
import statistics
import time
import urllib.request

import os
API = os.environ.get("API_BASE", "http://localhost:8000")
MODEL = "glm-5.3-flash"
PROMPTS = [
    ("Q&A",      "What are the main differences between TCP and UDP? Be concise.", 256),
    ("Code",     "Write a Python function that implements binary search on a sorted list. Include type hints and docstring.", 512),
    ("JSON",     "Generate a JSON array of 10 fictional employees with fields: name, age, department, salary, email, skills (array of 3). Output ONLY valid JSON, no explanation.", 1024),
    ("Math",     "What is 7823 * 4519? Show only the answer.", 64),
    ("LongCode", "Write a complete Python implementation of a red-black tree with insert, delete, search, and in-order traversal. Include all rotation methods.", 2048),
]
CONDITIONS = [
    ("t0/think-off", 0.0, False),
    ("t1/think-off", 1.0, False),
    ("t0/think-on",  0.0, True),
    ("t1/think-on",  1.0, True),
]


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


def chat(prompt, max_tokens, temp, thinking):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": 0.95 if temp > 0 else 1.0,
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    req = urllib.request.Request(
        API + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    return r, time.perf_counter() - t0


results = {}
for cname, temp, thinking in CONDITIONS:
    accs = {}
    for name, prompt, mt in PROMPTS:
        budget = mt * 2 if thinking else mt
        a0, d0 = scrape()
        try:
            r, el = chat(prompt, budget, temp, thinking)
        except Exception as e:
            print(f"  [{cname}][{name}] FAILED: {type(e).__name__}: {e}")
            continue
        a1, d1 = scrape()
        drafts = d1 - d0
        alen = (1 + (a1 - a0) / drafts) if drafts > 0 else float("nan")
        ct = r["usage"]["completion_tokens"]
        tps = ct / el
        accs[name] = alen
        print(f"  [{cname:13s}][{name:8s}] {ct:4d} tok {tps:5.1f} tok/s e2e | "
              f"accept {alen:.2f}/8 ({drafts:.0f} drafts)")
    results[cname] = accs

print("\n=== accept-length matrix (mean over categories) ===")
for cname, accs in results.items():
    vals = [v for v in accs.values() if v == v]
    row = "  ".join(f"{k}={v:.2f}" for k, v in accs.items())
    print(f"{cname:13s} mean={statistics.mean(vals):.2f}  |  {row}")
