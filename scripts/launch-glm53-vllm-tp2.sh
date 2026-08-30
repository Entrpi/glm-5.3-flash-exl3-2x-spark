#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash EXL3 + DFlash2, TP2 across two DGX Spark (GB10) boxes.
#
# Run the WORKER first (rank 1, ~25 s to join), then the HEAD (rank 0: API
# server + engine). The API serves on the head: http://<head-lan-ip>:8000.
# RELAUNCHING A LIVE CLUSTER: remove the head container BEFORE launching the
# new worker (a fresh worker rendezvouses with the old head's TCP store and
# dies of connection-reset when that head goes away). Then keep the worker->
# head gap under torch's 600 s rendezvous timeout.
# Community-derivative image: ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark
# (vLLM branch glm53-on-infernal + DFlash2 + ring draft-KV + EXL3 fused MoE
# baked; see docs/BUILD.md in the setup repo for full provenance).
#
# Per-box config lives in $HOME/.glm53-serve.env (written by install.sh);
# every value there uses the `: "${VAR:=...}"` idiom, so environment
# variables on the command line always win, e.g.:
#   KV_DTYPE=fp8_e4m3 ./launch-glm53-vllm-tp2.sh 0
NODE_RANK="${1:?usage: launch-glm53-vllm-tp2.sh <0|1>   (0=head/API, 1=worker)}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

GLM53_ENV="${GLM53_ENV:-$HOME/.glm53-serve.env}"
[[ -f "$GLM53_ENV" ]] && . "$GLM53_ENV"

# ---- topology (from .glm53-serve.env; defaults are the reference kit) ------
HEAD_RAIL_IP="${HEAD_RAIL_IP:-10.200.0.15}"      # head's IP on the 200GbE rail
WORKER_RAIL_IP="${WORKER_RAIL_IP:-10.200.0.33}"  # worker's IP on the same rail
NCCL_IF="${NCCL_IF:-enp1s0f0np0}"                # THIS box's rail interface
NCCL_HCA="${NCCL_HCA:-rocep1s0f0}"               # THIS box's RoCE HCA
NCCL_SUBNET="${NCCL_SUBNET:-10.200.0.0/24}"      # rail subnet for NCCL_IB_ADDR_RANGE
PORT="${PORT:-8000}"
MPORT="${MPORT:-29521}"

# ---- weights ---------------------------------------------------------------
# WEIGHTS_MODE=local : EXL3 weights on THIS box at $MODEL_HOST_PATH (default).
# WEIGHTS_MODE=nfs   : weights live on the WORKER; the worker exports them
#                      (containerized NFS, see install.sh) and the head mounts
#                      a docker NFS volume. This is the topology the reference
#                      kit runs in production (its head box lacks the disk).
#   CAUTION (measured on the reference kit): a HEAD process reading the
#   checkpoint from fast local NVMe can outrun GB10 unified-memory page-cache
#   reclaim and wedge the box into swap at ~90% of shard load. local mode is
#   the simpler default; if your head wedges during load, re-run install.sh
#   with --nfs. The drop-caches ritual below is required either way.
WEIGHTS_MODE="${WEIGHTS_MODE:-local}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/models/glm53-exl3}"
DFLASH_DIR="${DFLASH_DIR:-$HOME/models/glm53-dflash2}"
NFS_PORT="${NFS_PORT:-12049}"                    # worker's NFS export port (nfs mode)
VOL_NAME="${VOL_NAME:-exl3weights}"
MODEL_PATH="/models/glm53-exl3"

# ---- serving knobs (defaults = the validated production configuration) -----
IMAGE="${IMAGE:-ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v1-dflash2}"
NAME="${NAME:-vllm_glm53}"
MAX_LEN="${MAX_LEN:-524288}"             # 500k default bank (2026-08-29);
                                         # 131072 was the pre-long-context
                                         # default and still works with
                                         # KV_DTYPE= (bf16)
SPEC="${SPEC:-dflash}"                   # dflash (default) | none; MTP>0 overrides
MTP="${MTP:-0}"                          # fallback: MTP=4 SPEC=none
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"      # trained block size 8 = 1 bonus + 7 masks
EAGER="${EAGER:-0}"                      # 0 = CUDA graphs (validated); 1 = eager
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"  # skip the max-size multimodal
                                         # dummy profile (required at long
                                         # MAX_LEN on GB10 unified memory;
                                         # text profile still runs). 0 restores
                                         # profiling for short-context boots.
BLOCK_SIZE="${BLOCK_SIZE:-2304}"         # KV block; with fp8 KV vLLM auto-bumps
                                         # to 4608 to keep the KDA state page
                                         # equal to the attention page (boot
                                         # log: "Setting attention block size
                                         # to 4608"). 2304 is the bf16 parity
                                         # point; must satisfy
                                         # block %% (index_kpool*64).
KV_DTYPE="${KV_DTYPE-fp8_e4m3}"          # fp8_e4m3 (default 2026-08-29):
                                         # NOTE ${VAR-} not ${VAR:-}: an
                                         # explicitly EMPTY KV_DTYPE= selects
                                         # bf16; only unset gets the default.
                                         # 1,435,070-token pool @524k = 2.74
                                         # full banks; estonia 9/10, lavd
                                         # parity, math_500 88.0% pooled n=250
                                         # (bf16: 94%). empty = bf16: highest
                                         # math quality, pool ~520k tokens —
                                         # pair with MAX_LEN=131072.
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"   # empty -> 12.4e9 (1,435,070 tokens
                                         # @524k fp8 / 520,470 @131k bf16;
                                         # 5.25 GiB measured head floor);
                                         # "auto" -> vLLM budgeting;
                                         # 13.4e9 gave 567k bf16 but collapsed
                                         # the floor to 2.26 GiB — do NOT raise
                                         # without re-measuring memory floors.
MNBT="${MNBT:-8192}"                     # --max-num-batched-tokens: 112k-prompt
                                         # TTFT 93s->54.7s vs engine default
MM_CACHE_GB="${MM_CACHE_GB:-0.5}"        # mm processor cache (head-resident)
GMU="${GMU:-0.85}"                       # gpu-memory-utilization; 0.88+ risks
                                         # unified-memory swap on GB10
MAX_SEQS="${MAX_SEQS:-4}"                # 4 at the 524k default (2.74 banks);
                                         # 6 was the 131k-era value
CACHE_HOST_PATH="${CACHE_HOST_PATH:-$HOME/glm53-vllm-cache}"

if [[ "$KV_CACHE_MEMORY" == "auto" ]]; then
  KV_CACHE_MEMORY=""
elif [[ -z "$KV_CACHE_MEMORY" ]]; then
  KV_CACHE_MEMORY="12400000000"
fi

case "$NODE_RANK" in
  0) HOST_IP="$HEAD_RAIL_IP"; HEADLESS="" ;;
  1) HOST_IP="$WORKER_RAIL_IP"; HEADLESS="--headless" ;;
esac
ip -o addr show | grep -q "$HOST_IP/" || {
  echo "this host does not own $HOST_IP (rank $NODE_RANK; check HEAD_RAIL_IP/WORKER_RAIL_IP in $GLM53_ENV)" >&2
  exit 2
}

EXTRA_VOLS=()
EXTRA_ENVS=()
# Optional kernel-selection override (A/B tests without editing this script),
# e.g. VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel to step the
# MXFP8 draft GEMM ladder down to Marlin W8A16.
[[ -n "${VLLM_DISABLED_KERNELS:-}" ]] && EXTRA_ENVS+=(-e "VLLM_DISABLED_KERNELS=$VLLM_DISABLED_KERNELS")
# NVFP4 KV lane (KV_DTYPE=nvfp4_ds_mla): the rope-less 304 B/token record
# serves ONLY the dynamic per-token-scale mode; the engine refuses to boot
# without this env, so forward it whenever set.
[[ -n "${VLLM_NVFP4_MLA_DYNAMIC_SCALE:-}" ]] && EXTRA_ENVS+=(-e "VLLM_NVFP4_MLA_DYNAMIC_SCALE=$VLLM_NVFP4_MLA_DYNAMIC_SCALE")
# Interim persistent_topk override for pre-fix images (see fork 97f13931e):
# point at a built topk_fix.so inside the container to lift the 1M-declaration
# and drafterless-524k persistent_topk oversubscription.
[[ -n "${GLM53_TOPK_FIX_SO:-}" ]] && EXTRA_ENVS+=(-e "GLM53_TOPK_FIX_SO=$GLM53_TOPK_FIX_SO")
[[ -n "${VLLM_EXL3_STANDARD_FUSED:-}" ]] && EXTRA_ENVS+=(-e "VLLM_EXL3_STANDARD_FUSED=$VLLM_EXL3_STANDARD_FUSED")
if [[ "$WEIGHTS_MODE" == "local" || "$NODE_RANK" == "1" ]]; then
  test -f "$MODEL_HOST_PATH/config.json" || {
    echo "EXL3 weights not found at $MODEL_HOST_PATH (run install.sh, or set MODEL_HOST_PATH)" >&2
    exit 2
  }
  MODEL_VOL="$MODEL_HOST_PATH:$MODEL_PATH:ro"
else
  # nfs mode, head: mount the worker's containerized export (fsid=0 -> device=:/)
  docker volume inspect "$VOL_NAME" >/dev/null 2>&1 || docker volume create --driver local \
    --opt type=nfs --opt "o=addr=$WORKER_RAIL_IP,ro,vers=4.2,rsize=1048576,port=$NFS_PORT" \
    --opt device=:/ "$VOL_NAME" >/dev/null
  MODEL_VOL="$VOL_NAME:$MODEL_PATH:ro"
fi

SPEC_ARGS=()
if [[ "$MTP" != "0" ]]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
elif [[ "$SPEC" == "dflash" ]]; then
  test -f "$DFLASH_DIR/config.json" || {
    echo "DFlash2 draft weights not found at $DFLASH_DIR (run install.sh, or SPEC=none)" >&2
    exit 2
  }
  EXTRA_VOLS+=(-v "$DFLASH_DIR:/models/glm53-dflash2:ro")
  SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/glm53-dflash2\",\"num_speculative_tokens\":$DFLASH_TOKENS}")
fi
KV_ARGS=()
[[ -n "$KV_DTYPE" ]] && KV_ARGS+=(--kv-cache-dtype "$KV_DTYPE")
# Explicit attention backend (e.g. ATTN_BACKEND=B12X_MLA_SPARSE for the
# GLM_NEXT 528 B/token lane, which also needs KV_DTYPE=fp8_ds_mla). Empty =
# the fork's auto-selection (FLASHINFER_MLA_SPARSE_SM90 on these boxes).
ATTN_BACKEND="${ATTN_BACKEND:-}"
[[ -n "$ATTN_BACKEND" ]] && KV_ARGS+=(--attention-backend "$ATTN_BACKEND")
# Layers exempt from KV quantization. The GLM_NEXT fp8_ds_mla lane needs the
# DFlash2 draft ring-KV layers on bf16 (the packed 528 B record is
# target-MLA-only): use KV_SKIP_LAYERS=sliding_window — the draft ring is
# sliding-window-typed, and the single token avoids the CLI's list parsing
# (a comma-joined index string arrives as one unmatched element).
KV_SKIP_LAYERS="${KV_SKIP_LAYERS:-}"
[[ -n "$KV_SKIP_LAYERS" ]] && KV_ARGS+=(--kv-cache-dtype-skip-layers "$KV_SKIP_LAYERS")
[[ -n "$KV_CACHE_MEMORY" ]] && KV_ARGS+=(--kv-cache-memory "$KV_CACHE_MEMORY")
EAGER_ARGS=()
[[ "$EAGER" != "0" ]] && EAGER_ARGS=(--enforce-eager)
[[ "$SKIP_MM_PROFILING" != "0" ]] && EAGER_ARGS+=(--skip-mm-profiling)
MNBT_ARGS=()
[[ -n "$MNBT" ]] && MNBT_ARGS=(--max-num-batched-tokens "$MNBT")

# Hotfix hooks: any file under $HOME/glm53-hotfix (mirroring the vllm package
# tree) or $HOME/glm53-hotfix-fi (flashinfer tree) is bind-mounted over the
# installed package — fast community debugging without an image rebuild. The
# dirs do not exist in a normal install; REMOVE them (never empty in place)
# once a fix is folded into an image.
HOTFIX_DIR="$HOME/glm53-hotfix"
if [[ -d "$HOTFIX_DIR" ]]; then
  while IFS= read -r -d '' f; do
    rel="${f#"$HOTFIX_DIR"/}"
    EXTRA_VOLS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro")
  done < <(find "$HOTFIX_DIR" -type f -print0)
fi
FI_HOTFIX_DIR="$HOME/glm53-hotfix-fi"
if [[ -d "$FI_HOTFIX_DIR" ]]; then
  while IFS= read -r -d '' f; do
    rel="${f#"$FI_HOTFIX_DIR"/}"
    EXTRA_VOLS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/flashinfer/$rel:ro")
  done < <(find "$FI_HOTFIX_DIR" -type f -print0)
fi
# b12x is bumped as a whole package (kernel contracts span many files), so
# this hook binds the directory, not per-file: put a complete b12x/ package
# tree at $HOME/glm53-hotfix-b12x (the package dir itself, containing
# __init__.py). Same rule as the others: REMOVE the dir once folded into an
# image.
B12X_HOTFIX_DIR="$HOME/glm53-hotfix-b12x"
if [[ -d "$B12X_HOTFIX_DIR" ]]; then
  EXTRA_VOLS+=(-v "$B12X_HOTFIX_DIR:/usr/local/lib/python3.12/dist-packages/b12x:ro")
fi

# Persist JIT compile caches (Triton, FlashInfer, b12x CuTeDSL, vLLM
# torch.compile) across container recreates — hash-keyed, so stale entries
# are inert. Without this every relaunch re-JITs from scratch.
mkdir -p "$CACHE_HOST_PATH" \
  "$CACHE_HOST_PATH/jit/triton" "$CACHE_HOST_PATH/jit/flashinfer" \
  "$CACHE_HOST_PATH/jit/b12x" "$CACHE_HOST_PATH/jit/vllm"
docker rm -f "$NAME" 2>/dev/null || true

# GB10 pre-launch ritual — a hot page cache at model-load time wedges the box
# into swap (unified-memory starvation). Root via privileged docker; falls
# back to passwordless sudo where available.
sync
docker run --rm --privileged alpine sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" >/dev/null 2>&1 \
  || { echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; }

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_VOL" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$CACHE_HOST_PATH/jit/triton:/root/.triton" \
  -v "$CACHE_HOST_PATH/jit/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_HOST_PATH/jit/b12x:/root/.cache/b12x" \
  -v "$CACHE_HOST_PATH/jit/vllm:/root/.cache/vllm" \
  "${EXTRA_VOLS[@]}" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  "${EXTRA_ENVS[@]}" \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$NCCL_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="$NCCL_SUBNET" \
  -e NCCL_SOCKET_IFNAME="$NCCL_IF" -e GLOO_SOCKET_IFNAME="$NCCL_IF" \
  -e TP_SOCKET_IFNAME="$NCCL_IF" -e MN_IF_NAME="$NCCL_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    vllm serve "$MODEL_PATH" \
    --served-model-name glm-5.3-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code --quantization exl3 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GMU" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" --block-size "$BLOCK_SIZE" --mm-processor-cache-gb "$MM_CACHE_GB" \
    "${MNBT_ARGS[@]}" "${SPEC_ARGS[@]}" "${KV_ARGS[@]}" \
    "${EAGER_ARGS[@]}" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_RAIL_IP" --master-port "$MPORT" \
    $HEADLESS

echo "launched $NAME rank=$NODE_RANK host=$HOST_IP image=$IMAGE weights=$WEIGHTS_MODE kv=${KV_CACHE_MEMORY:-auto}${KV_DTYPE:+/$KV_DTYPE}${ATTN_BACKEND:+ attn=$ATTN_BACKEND} spec=${SPEC}${MTP:+ mtp=$MTP} mnbt=${MNBT:-default}"
