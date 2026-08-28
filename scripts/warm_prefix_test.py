"""Warm-prefix drafting test: does DFlash2 keep drafting after prefix-cache
restores of a long (multi-block) shared prefix?

Three shapes, per-request /metrics deltas (server must be idle):
  A cold:  long doc (+~3 blocks) + question 1        -> expect normal drafting
  B warm:  same doc + question 2 (shared prefix hit) -> drafts==0 means disabled
  C turn2: [doc+q1, answer_A, question 3]            -> realistic agentic case
"""

import json
import os
import time
import urllib.request

API = os.environ.get("API_BASE", "http://localhost:8000")
MODEL = "glm-5.3-flash"

sections = []
for i in range(1, 16):
    sections.append(
        f"Section {i}: The {i}th subsystem of the Meridian pipeline handles "
        f"stage-{i} dataflow. It was commissioned in 20{10 + i} by the team "
        f"led by engineer number {i * 37}. Its throughput is {i * 113} units "
        f"per cycle, with a latency budget of {i * 7} milliseconds and a "
        f"failover window of {i * 3} seconds. The subsystem depends on "
        f"section {max(1, i - 1)} for upstream batching and feeds section "
        f"{i + 1} downstream. Notable incidents include the 20{12 + i} "
        f"overflow event, resolved by widening the ring buffer to {i * 256} "
        f"entries, and a clock-skew regression fixed in firmware {i}.{i}.{i}. "
        f"Operational notes: the {i}th maintenance crew rotates every "
        f"{i + 2} weeks, and calibration drift stays below {i} ppm. " * 4
    )
DOC = "\n\n".join(sections)


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


def chat(messages, max_tokens):
    body = json.dumps({
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        API + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    el = time.perf_counter() - t0
    u = r["usage"]
    return r["choices"][0]["message"]["content"], u, el


def run(name, messages, max_tokens=192):
    a0, d0 = scrape()
    content, usage, el = chat(messages, max_tokens)
    a1, d1 = scrape()
    drafts = d1 - d0
    acc = a1 - a0
    alen = (1 + acc / drafts) if drafts > 0 else 0.0
    print(f"{name}: prompt_tokens={usage['prompt_tokens']} "
          f"completion={usage['completion_tokens']} wall={el:.2f}s | "
          f"drafts={drafts:.0f} accepted={acc:.0f} accept_len={alen:.2f}/8"
          + ("  << DRAFTING DISABLED" if drafts == 0 else ""))
    return content


q1 = DOC + "\n\nQuestion 1: Summarize section 7 in two sentences."
answer_a = run("A cold  ", [{"role": "user", "content": q1}])

q2 = DOC + "\n\nQuestion 2: What throughput does section 12 achieve, and which firmware fixed its clock-skew regression?"
run("B warm  ", [{"role": "user", "content": q2}])

run("C turn2 ", [
    {"role": "user", "content": q1},
    {"role": "assistant", "content": answer_a},
    {"role": "user", "content": "Question 3: Which section has the largest failover window and what is it?"},
])
