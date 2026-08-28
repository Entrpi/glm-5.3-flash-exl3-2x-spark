"""~100k-context smoke on the DFlash2 server.

Single streaming request with a ~100k-token synthetic document + question.
Measures TTFT (first streamed token), decode tok/s (first->last token),
per-request accept delta, and answer sanity (the doc plants retrievable
facts). Confirms the draft's sliding-window KV behaves at real depth and the
329k pool carries a 100k+ single stream.
"""

import json
import os
import time
import urllib.request

API = os.environ.get("API_BASE", "http://localhost:8000")
MODEL = "glm-5.3-flash"
NSECT = int(os.environ.get("NSECT", "135"))  # 135 ~= 112k tok; 40 ~= 33k

sections = []
for i in range(1, NSECT + 1):
    sections.append(
        f"Chapter {i}. The {i}th expedition of the Halcyon survey charted "
        f"basin {i * 13} at a depth of {i * 41} meters, logging {i * 7} "
        f"specimen classes and a mean salinity of {30 + (i % 9)}.{i % 10} "
        f"PSU. Lead researcher badge {i * 199} filed report R-{i:04d}, "
        f"noting thermal vents at coordinates ({i * 3}.{i % 7}, "
        f"{i * 5}.{i % 4}) and a sonar anomaly classified level {i % 6}. "
        f"Supply manifests listed {i * 11} crates, {i * 2} submersible "
        f"sorties, and {i % 5 + 1} equipment failures, the worst being the "
        f"winch jam of day {i % 28 + 1}. Follow-up actions were assigned to "
        f"team {chr(65 + i % 26)} with a deadline {i % 12 + 1} months out. " * 6
    )
DOC = "\n\n".join(sections)
TARGET = 121 if NSECT >= 121 else max(1, NSECT - 7)
QUESTION = (
    f"\n\nQuestion: What depth did expedition number {TARGET} chart its "
    "basin at, what was its report number, and how many specimen classes "
    "did it log? Answer from the document only, in one sentence."
)


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


body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": DOC + QUESTION}],
    "temperature": 0, "max_tokens": 256, "seed": 12345,
    "stream": True, "stream_options": {"include_usage": True},
}).encode()

a0, d0 = scrape()
req = urllib.request.Request(
    API + "/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
t_first = t_last = None
completion_tokens = prompt_tokens = 0
text = []
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        u = obj.get("usage")
        if u:
            completion_tokens = u.get("completion_tokens", completion_tokens)
            prompt_tokens = u.get("prompt_tokens", prompt_tokens)
        ch = obj.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            text.append(ch[0]["delta"]["content"])
a1, d1 = scrape()

ttft = (t_first - t0) if t_first else float("nan")
decode = (completion_tokens - 1) / (t_last - t_first) if t_last and t_last > t_first else float("nan")
drafts = d1 - d0
alen = 1 + (a1 - a0) / drafts if drafts > 0 else 0.0
answer = "".join(text)
print(f"prompt_tokens={prompt_tokens} completion={completion_tokens}")
print(f"TTFT={ttft:.2f}s decode={decode:.2f} tok/s | drafts={drafts:.0f} "
      f"accept={alen:.2f}/8" + ("  << NO DRAFTING" if drafts == 0 else ""))
print(f"expected facts: depth {TARGET * 41} m, report R-{TARGET:04d}, "
      f"{TARGET * 7} specimen classes")
print(f"answer: {answer[-400:]}")
