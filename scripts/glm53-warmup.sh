#!/usr/bin/env bash
# Post-boot shape warmup for the GLM-5.3-Flash DFlash2 serve (idea carried
# over from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks boot-shape-warmup).
# The draft's CUDA graphs are captured at boot; what stays lazy on our stack
# is prefill/kpool JIT, the temp>0 sampler path, and batch decode shapes —
# so the first real client eats ~7s of TTFT. Fire those shapes now instead.
# JIT caches persist across relaunches (launch-script /cache/jit mounts), so
# this mainly pays on the first boot of a new image.
#
# Usage: API_BASE=http://localhost:8000 bash glm53-warmup.sh
set -u
API="${API_BASE:-http://localhost:8000}"
MODEL="${MODEL:-glm-5.3-flash}"
TIMEOUT="${WARMUP_REQ_TIMEOUT:-300}"

req() { # req <max_tokens> <temperature> <thinking> <prompt>
  curl -s -m "$TIMEOUT" "$API/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$4"'"}],"max_tokens":'"$1"',"temperature":'"$2"',"chat_template_kwargs":{"enable_thinking":'"$3"'}}' \
    >/dev/null 2>&1
}

echo "warmup: waiting for API at $API"
t0=$(date +%s)
until curl -sf -m5 "$API/v1/models" >/dev/null 2>&1; do
  sleep 10
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "warmup: API never came up"; exit 1; }
done

t0=$(date +%s)
echo "warmup: c1 greedy (decode + parser)"
req 64 0 false "Count from 1 to 30. Output only the numbers."
echo "warmup: c1 sampled (Gumbel/temp path, thinking on)"
req 48 0.8 true "Name three prime numbers."
echo "warmup: c4 batch decode shapes"
for i in 1 2 3 4; do
  req 96 0 false "Explain in two sentences why the sky is blue. Variant $i." &
done
wait
echo "warmup: ~8k-token prefill (chunked prefill + kpool shapes)"
LONG=$(python3 -c "print(('The relay station processes telemetry frames in fixed windows, applying vector quantization to each channel before forwarding summaries to the archive tier. ' * 320).replace('\"',''))")
req 32 0 false "$LONG Summarize the above in one sentence."
echo "warmup: done in $(( $(date +%s) - t0 ))s"
