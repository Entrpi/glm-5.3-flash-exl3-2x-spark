#!/bin/bash
# Phase 3 image self-containment gauntlet (no GPU needed).
# Usage: bash check_image_phase3.sh [image]  (default vllm-node-glm53:v2-glmnext)
set -u
IMG="${1:-vllm-node-glm53:v2.1-glmnext}"
DP=/usr/local/lib/python3.12/dist-packages
pass=0; fail=0
chk() { # chk <label> <shell snippet that exits 0 on pass>
  if docker run --rm --entrypoint sh "$IMG" -c "$2" >/dev/null 2>&1; then
    echo "PASS: $1"; pass=$((pass+1))
  else
    echo "FAIL: $1"; fail=$((fail+1))
  fi
}
EXPECT_COMMIT="${EXPECT_COMMIT:-1d220461f}"
chk "vllm version stamps $EXPECT_COMMIT" "pip show vllm | grep -q g$EXPECT_COMMIT"
chk "mm jinja has thinking gate"         "grep -q thinking_enabled $DP/vllm/transformers_utils/chat_templates/template_glm5next_mm.jinja"
chk "glm5next mm jinja packaged"        "test -f $DP/vllm/transformers_utils/chat_templates/template_glm5next_mm.jinja"
chk "chat-template repair hook"         "grep -q glm5next_mm $DP/vllm/entrypoints/renderers/hf.py || grep -rq glm5next_mm $DP/vllm/transformers_utils/chat_templates/registry.py"
chk "DFlash2 model file"                "test -f $DP/vllm/model_executor/models/qwen3_dflash.py"
chk "DFlash2 speculator + ring remap"   "grep -q ring_remap_draft_block_tables $DP/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
chk "DFlashSWASpec in kv_cache_interface" "grep -q DFlashSWASpec $DP/vllm/v1/kv_cache_interface.py"
chk "ring lane in kv_cache_utils"       "grep -q _dflash_ring_pages $DP/vllm/v1/core/kv_cache_utils.py"
chk "VLLM_DFLASH_KV_RING env"           "grep -q VLLM_DFLASH_KV_RING $DP/vllm/envs.py"
chk "exl3ext baked at /exl3ext"         "test -f /exl3ext/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"
chk "VLLM_EXL3_EXT_PATH env baked"      "test \"\$VLLM_EXL3_EXT_PATH\" = /exl3ext"
chk "b12x = sparkinfer exl3 lane"       "grep -rq exl3_trellis_mcg $DP/b12x/"

fchk() { # fchk <label> <container path> <host path> — exact content match
  a=$(docker run --rm --entrypoint sh "$IMG" -c "md5sum $2 2>/dev/null" | awk '{print $1}')
  b=$(md5sum "$3" 2>/dev/null | awk '{print $1}')
  if [ -n "$a" ] && [ "$a" = "$b" ]; then
    echo "PASS: $1"; pass=$((pass+1))
  else
    echo "FAIL: $1 (image=$a host=$b)"; fail=$((fail+1))
  fi
}
# Staged-patch comparisons only run where the build staging dir exists
# (the reference build box); elsewhere the grep-level checks above suffice.
FIN="${FIN:-$HOME/phase3-finalize}"
if [ ! -d "$FIN" ]; then
  echo "---"
  echo "image=$IMG pass=$pass fail=$fail (staging-dir checks skipped: no $FIN)"
  exit $fail
fi
fchk "FI fp8-MLA _core.py = staged patch" "$DP/flashinfer/mla/_core.py" "$FIN/fi-patches/mla/_core.py"
fchk "FI mla.cuh = staged patch" "$DP/flashinfer/data/include/flashinfer/attention/mla.cuh" "$FIN/fi-patches/data/include/flashinfer/attention/mla.cuh"

# b12x whole-tree hash (py files only; order-stable) vs the pinned
# Entrpi/sparkinfer-glmrt glm-next-backport @ 3ce6115 tree — computed as
#   cd b12x && LC_ALL=C find . -name '*.py' -type f | LC_ALL=C sort \
#     | xargs md5sum | md5sum
# Self-contained on purpose: v2 bakes b12x in the base build, so there is
# no staged host tree to compare against any more.
tb="${B12X_TREE_MD5:-b338deafc04a92ac4099daab2c8da123}"
ti=$(docker run --rm --entrypoint sh "$IMG" -c "cd $DP/b12x && LC_ALL=C find . -name '*.py' -type f | LC_ALL=C sort | xargs md5sum | md5sum" | awk '{print $1}')
if [ "$tb" = "$ti" ]; then echo "PASS: b12x tree = sparkinfer glm-next-backport @3ce6115"; pass=$((pass+1))
else echo "FAIL: b12x tree mismatch (image=$ti expected=$tb)"; fail=$((fail+1)); fi
echo "---"
echo "image=$IMG pass=$pass fail=$fail"
exit $fail
