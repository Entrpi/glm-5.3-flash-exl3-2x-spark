#!/usr/bin/env bash
# Run the DFlash2 + ring + glm5-grouping + chat-template unit tests inside
# the pulled image (self-contained: no overlays). --gpus all is required —
# the ring-kernel and glm5-config tests need a visible GPU, so run this
# between teardown and launch, never while serving.
#
# The tests live in the vLLM branch repo; a shallow sparse clone of tests/
# is cached under /tmp and reused.
set -euo pipefail
IMG="${IMG:-ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v1-dflash2}"
BRANCH_REPO="${BRANCH_REPO:-https://github.com/Entrpi/vllm-glm-5.3-flash-spark}"
TESTS_DIR="${TESTS_DIR:-/tmp/glm53-vllm-tests}"

if [ ! -d "$TESTS_DIR/tests" ]; then
  rm -rf "$TESTS_DIR"
  git clone --quiet --depth 1 --filter=blob:none --sparse "$BRANCH_REPO" "$TESTS_DIR"
  git -C "$TESTS_DIR" sparse-checkout set tests
fi

docker run --rm --gpus all \
  -v "$TESTS_DIR/tests:/ws/tests:ro" \
  -w /ws --entrypoint bash \
  "$IMG" \
  -c 'pip install -q pytest tblib >/dev/null 2>&1
python3 -m pytest \
  tests/v1/spec_decode/test_dflash2.py \
  tests/v1/spec_decode/test_dflash_causality.py \
  tests/models/test_glm5next_dflash_capture.py \
  "tests/test_config.py::test_dflash2_draft_forces_v2_model_runner" \
  "tests/v1/core/test_kv_cache_utils.py::test_glm5_grouping_with_dflash_draft_layers" \
  "tests/v1/core/test_kv_cache_utils.py::test_glm5_dflash_ring_backed_draft_kv" \
  tests/transformers_utils/test_glm5next_chat_template.py \
  -q --no-header 2>&1 | tail -5'
