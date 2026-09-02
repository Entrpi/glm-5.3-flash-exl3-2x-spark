#!/usr/bin/env python3
"""Single-request decode speed by workload, with the spread and the drafter's
acceptance, against a running server. Stdlib only.

  python3 scripts/bench_workloads.py [runs=5] [max_tokens=400] [phase,phase,...]

Each phase sends the same prompt `runs` times (temperature 0, thinking off,
natural stop allowed) and reports the median and min-max of decode tok/s,
plus accepted tokens per step and steps/s from /metrics deltas. Greedy
continuations are not bit-repeatable on a tensor-parallel MoE, so a single
run's tok/s is an acceptance draw; quote the median with the spread.
"""
import json, os, re, sys, time, urllib.request

BASE = os.environ.get("API_BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "glm-5.3-flash")
PHASES = {
    "prose": "Write a detailed step-by-step explanation of how a hash map works, "
             "including collision handling, resizing, and time complexity. Be thorough.",
    "code": "Write a Python module with a class LRUCache (get/put, O(1), with a doubly linked list "
            "and a dict), full docstrings, type hints, and a small pytest test file at the end. Code only.",
    "json": "Produce a JSON array of 20 fictional employees. Each object has fields id, name, "
            "department, salary, start_date (ISO), skills (array of 3 strings), and address "
            "(object with street, city, country). Output only the JSON.",
    "math": "Solve step by step: a train leaves at 9:15 travelling 72 km/h; a second train leaves "
            "the same station at 9:45 at 96 km/h on the same track. When and where does it catch up? "
            "Then generalise to speeds v1 < v2 and head start t. Show all algebra.",
    "structured": "Count from 1 to 200, one number per line, nothing else.",
}


def metrics():
    txt = urllib.request.urlopen(f"{BASE}/metrics", timeout=10).read().decode()
    out = {}
    for key, pat in (("steps", r"vllm:spec_decode_num_drafts_total\{[^}]*\} ([0-9.e+]+)"),
                     ("accepted", r"vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} ([0-9.e+]+)")):
        m = re.search(pat, txt)
        out[key] = float(m.group(1)) if m else None
    return out


def one(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None; finish = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")) and t_first is None:
                    t_first = time.time()
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.time()
    n = (usage or {}).get("completion_tokens") or 0
    dec = (n - 1) / (t_end - t_first) if t_first and n > 1 else float("nan")
    return dict(ttft=(t_first or t_end) - t0, decode=dec, out=n, finish=finish, wall=t_end - t_first if t_first else 0)


def med(xs):
    xs = sorted(xs); m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    phases = sys.argv[3].split(",") if len(sys.argv) > 3 else list(PHASES)
    one(PHASES["prose"], 32)  # warm
    have_spec = metrics()["steps"] is not None
    print(f"runs={runs} max_tokens={max_tokens} spec_metrics={have_spec}")
    rows = []
    for ph in phases:
        decs, accs, sps, outs, ttfts = [], [], [], [], []
        for i in range(runs):
            m0 = metrics(); r = one(PHASES[ph], max_tokens); m1 = metrics()
            decs.append(r["decode"]); outs.append(r["out"]); ttfts.append(r["ttft"])
            if have_spec and m1["steps"] and m0["steps"] is not None:
                st = m1["steps"] - m0["steps"]
                if st > 0:
                    accs.append((m1["accepted"] - m0["accepted"]) / st)
                    sps.append(st / r["wall"] if r["wall"] else float("nan"))
            print(f"  {ph} run{i+1}: decode={r['decode']:.1f} tok/s out={r['out']} finish={r['finish']} ttft={r['ttft']:.2f}s"
                  + (f" accepted/step={accs[-1]:.2f} steps/s={sps[-1]:.2f}" if accs and len(accs) == i + 1 else ""), flush=True)
        rows.append((ph, med(decs), min(decs), max(decs), med(accs) if accs else float("nan"),
                     med(sps) if sps else float("nan"), med(outs), med(ttfts)))
    print("\nphase        median tok/s   min-max        accepted/step  steps/s  out_tokens  ttft")
    for ph, m, lo, hi, acc, sp, o, tt in rows:
        print(f"{ph:<12} {m:>8.1f}       {lo:>5.1f}-{hi:<5.1f}     {acc:>6.2f}       {sp:>5.2f}    {o:>6.0f}     {tt:.2f}s")


if __name__ == "__main__":
    main()
