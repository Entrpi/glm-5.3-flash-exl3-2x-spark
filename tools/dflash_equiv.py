#!/usr/bin/env python3
"""Greedy token-equivalence harness for speculative-decoding losslessness.

Usage:
  python3 dflash_equiv.py dump <label>          # query the running server, save outputs
  python3 dflash_equiv.py compare <a> <b>       # diff two dumps token-for-token

Run `dump baseline` against the target-only (or MTP) server, relaunch with
SPEC=dflash, run `dump dflash`, then `compare baseline dflash`. Greedy
losslessness means every case matches exactly (content, finish reason, token
count). Covers: short EOS-bounded answers, long generations spanning many
blocks, prefix-cache hit (same prompt twice), and a concurrent batch.
"""

import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = os.environ.get("API", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "glm-5.3-flash")
OUTDIR = os.environ.get("OUTDIR", os.path.expanduser("~/dflash_equiv"))

CASES = [
    ("short_eos", "What is 2+2? Answer with just the number.", 64),
    ("mid_math", "Compute 17 * 23 step by step, then state the answer.", 256),
    ("long_prose", "Write a detailed 400-word explanation of how tides work.", 640),
    ("code", "Write a Python function that merges two sorted lists. Code only.", 384),
    ("list_gen", "List the first 25 prime numbers, comma separated.", 256),
    ("prefix_repeat", "Compute 17 * 23 step by step, then state the answer.", 256),
    # Exercises the draft's non-causal sliding window: generation must run
    # well past 2048 tokens of draft KV.
    (
        "very_long",
        "Write a thorough, chapter-structured 2500-word essay on the history "
        "of numerical linear algebra, from Gauss to GPUs.",
        3072,
    ),
]
CONCURRENT_CASE = ("concurrent", "Count from 1 to 40, one number per line.", 320)


def query(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": 12345,
        }
    ).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.load(r)
    choice = out["choices"][0]
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "completion_tokens": out["usage"]["completion_tokens"],
    }


def dump(label: str) -> None:
    results = {}
    for name, prompt, max_tokens in CASES:
        results[name] = query(prompt, max_tokens)
        print(f"  {name}: {results[name]['completion_tokens']} tokens")
    name, prompt, max_tokens = CONCURRENT_CASE
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(query, prompt, max_tokens) for _ in range(5)]
        results[name] = [f.result() for f in futs]
    print(f"  {name}: {[r['completion_tokens'] for r in results[name]]} tokens x5")
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{label}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {path}")


def compare(a: str, b: str) -> None:
    with open(os.path.join(OUTDIR, f"{a}.json")) as f:
        da = json.load(f)
    with open(os.path.join(OUTDIR, f"{b}.json")) as f:
        db = json.load(f)
    failures = 0
    for name in da:
        va, vb = da[name], db[name]
        pairs = zip(va, vb) if isinstance(va, list) else [(va, vb)]
        for i, (xa, xb) in enumerate(pairs):
            tag = f"{name}[{i}]" if isinstance(va, list) else name
            if xa == xb:
                print(f"  MATCH {tag} ({xa['completion_tokens']} tokens)")
                continue
            failures += 1
            print(f"  DIFF  {tag}:")
            for k in ("finish_reason", "completion_tokens"):
                if xa[k] != xb[k]:
                    print(f"        {k}: {xa[k]} vs {xb[k]}")
            ca, cb = xa["content"], xb["content"]
            if ca != cb:
                pos = next(
                    (j for j, (p, q) in enumerate(zip(ca, cb)) if p != q),
                    min(len(ca), len(cb)),
                )
                print(f"        content diverges at char {pos}:")
                print(f"          {a}: ...{ca[max(0, pos - 40) : pos + 40]!r}")
                print(f"          {b}: ...{cb[max(0, pos - 40) : pos + 40]!r}")
    print(("EQUIVALENT" if failures == 0 else f"NOT EQUIVALENT: {failures} diffs"))
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
