#!/usr/bin/env bash
# GLM-5.3-Flash (EXL3 4bpw + DFlash2 speculative decode) on 2x NVIDIA DGX
# Spark — one-shot installer. Run ON THE HEAD BOX (the one that will serve
# the API), with passwordless SSH to the worker:
#
#   git clone https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark
#   cd glm-5.3-flash-exl3-2x-spark
#   cp .env.example .env   # edit WORKER_LAN_IP + rail IPs/interfaces
#   ./install.sh
#
# What this does (every step idempotent — safe to re-run):
#   1. verify both hosts (arch, GB10, docker+GPU, disk, rail, SSH)
#   2. pull the serving image on the head; ship to the worker if needed
#   3. download weights (EXL3 ~176 GiB; DFlash2 drafter ~2.3 GiB) per the
#      chosen topology (local-both default, --nfs alternative)
#   4. install the launch/warmup scripts + per-box serve config
#   5. launch worker then head, wait for health, warm JIT shapes, smoke test
#
# The script makes NO changes outside of:
#   - the docker image/volume/container namespace (image, vllm_glm53,
#     nfs-exl3 in --nfs mode, exl3weights volume)
#   - $MODEL_HOST_PATH, $DFLASH_DIR, $HOME/glm53-vllm-cache (both boxes)
#   - $HOME/.glm53-serve.env, $HOME/launch-glm53-vllm-tp2.sh,
#     $HOME/glm53-warmup.sh (both boxes)
#
# This recipe is a community derivative. It layers on top of:
#   zai-org/GLM-5.3-Flash (model) - brandonmusic EXL3/TR3 4bpw (quant)
#   incoai GLM-5.3-Flash-DFlash2 (drafter, CC BY-NC-ND, fetched from source)
#   vLLM + local-inference-lab fork lineage - eugr's spark-vllm-docker build
#   turboderp exllamav3 - tpurtell sparkinfer-glmrt (b12x) - FlashInfer
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- §0 config + flags ------------------------------------------------
[ -f "$SCRIPT_DIR/.env" ] || { [ -f "$SCRIPT_DIR/.env.example" ] && cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env" && echo "NOTE: created .env from .env.example — edit it and re-run if these defaults are wrong."; }
[ -f "$SCRIPT_DIR/.env" ] && . "$SCRIPT_DIR/.env"

WORKER_LAN_IP="${WORKER_LAN_IP:-}"
WORKER_USER="${WORKER_USER:-}"
HEAD_RAIL_IP="${HEAD_RAIL_IP:-10.200.0.1}"
WORKER_RAIL_IP="${WORKER_RAIL_IP:-10.200.0.2}"
HEAD_NCCL_IF="${HEAD_NCCL_IF:-enp1s0f0np0}"
HEAD_NCCL_HCA="${HEAD_NCCL_HCA:-rocep1s0f0}"
WORKER_NCCL_IF="${WORKER_NCCL_IF:-enp1s0f0np0}"
WORKER_NCCL_HCA="${WORKER_NCCL_HCA:-rocep1s0f0}"
NCCL_SUBNET="${NCCL_SUBNET:-10.200.0.0/24}"
WEIGHTS_MODE="${WEIGHTS_MODE:-}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/models/glm53-exl3}"
DFLASH_DIR="${DFLASH_DIR:-$HOME/models/glm53-dflash2}"
EXL3_REPO="${EXL3_REPO:-brandonmusic/GLM-5.3-Flash-tr3-4bpw}"
EXL3_REPO_FALLBACK="${EXL3_REPO_FALLBACK:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}"
DFLASH_REPO="${DFLASH_REPO:-incoai/GLM-5.3-Flash-DFlash2}"
IMAGE="${IMAGE:-ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v1-dflash2}"
NFS_PORT="${NFS_PORT:-12049}"
PORT="${PORT:-8000}"

# --- serving knobs (forwarded to ~/.glm53-serve.env by write_env) ---
MPORT="${MPORT:-29521}"
MAX_LEN="${MAX_LEN:-524288}"
SPEC="${SPEC:-dflash}"
MTP="${MTP:-0}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
EAGER="${EAGER:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
KV_DTYPE="${KV_DTYPE-fp8_e4m3}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
KV_SKIP_LAYERS="${KV_SKIP_LAYERS:-}"
ATTN_BACKEND="${ATTN_BACKEND:-}"
MNBT="${MNBT:-8192}"
GMU="${GMU:-0.85}"
MAX_SEQS="${MAX_SEQS:-4}"
MM_CACHE_GB="${MM_CACHE_GB:-0.5}"

SKIP_PULL=0 SKIP_DOWNLOAD=0 NO_START=0 FORCE=0
usage() {
  cat <<'EOF'
Usage: ./install.sh [--nfs|--local-both] [--skip-pull] [--skip-download]
                    [--no-start] [--force] [--help]

  --nfs           weights only on the worker, NFS-exported to the head
                  (the reference kit's validated-production topology)
  --local-both    weights on both boxes (default; simplest, ~176 GiB/box.
                  If the head swap-wedges at ~90% of load, re-run --nfs)
  --skip-pull     keep the local image (no GHCR pull)
  --skip-download weights already present on the right boxes
  --no-start      prepare everything but do not launch
  --force         downgrade hardware-check failures to warnings

Env-var equivalents live in .env (copied from .env.example on first run).
EOF
}
for arg in "$@"; do
  case "$arg" in
    --nfs) WEIGHTS_MODE=nfs ;;
    --local-both) WEIGHTS_MODE=local ;;
    --skip-pull) SKIP_PULL=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    --no-start) NO_START=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $arg"; usage; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

[ -n "$WORKER_LAN_IP" ] || die "WORKER_LAN_IP is not set — edit .env (copied from .env.example) and re-run."
WORKER="${WORKER_USER:+$WORKER_USER@}$WORKER_LAN_IP"
WSSH() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }

if [ -z "$WEIGHTS_MODE" ]; then
  if [ -t 0 ]; then
    read -r -p "Weights topology: [L]ocal on both boxes (default) or [n]fs from the worker? " ans
    case "${ans:-l}" in n|N|nfs) WEIGHTS_MODE=nfs ;; *) WEIGHTS_MODE=local ;; esac
  else
    WEIGHTS_MODE=local
  fi
fi
log "topology: WEIGHTS_MODE=$WEIGHTS_MODE  head=$HEAD_RAIL_IP  worker=$WORKER ($WORKER_RAIL_IP)"

# ---------- §1 verify hosts --------------------------------------------------
hw_check() { # hw_check <where> <runner...>
  local where=$1; shift
  local run=("$@")
  "${run[@]}" 'uname -m' | grep -q aarch64 || { [ "$FORCE" = 1 ] && warn "$where: not aarch64" || die "$where is not aarch64 (--force to override)"; }
  "${run[@]}" 'command -v docker >/dev/null' || die "$where: docker not found"
  "${run[@]}" 'command -v nvidia-smi >/dev/null' || die "$where: nvidia-smi not found"
  "${run[@]}" 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader' | grep -Eq 'GB10|12\.1|Spark' \
    || { [ "$FORCE" = 1 ] && warn "$where: GPU is not a GB10/12.1" || die "$where: GPU is not a GB10 (--force to override)"; }
  "${run[@]}" 'docker run --rm --gpus all --entrypoint true alpine >/dev/null 2>&1 || docker run --rm --gpus all --entrypoint true ubuntu:24.04 >/dev/null 2>&1' \
    || warn "$where: docker --gpus all probe failed (may be fine if no small image is cached; the real pull comes next)"
}
disk_free_g() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc 0-9; }

verify_hosts() {
  log "verifying head"
  hw_check head bash -c
  log "verifying worker over SSH ($WORKER)"
  WSSH true || die "passwordless SSH to $WORKER failed (ssh-copy-id first)"
  hw_check worker WSSH
  ping -c1 -W2 "$WORKER_RAIL_IP" >/dev/null 2>&1 || warn "head cannot ping $WORKER_RAIL_IP — check the 200GbE rail cabling/addresses (NCCL will fail without it)"
  local need_head=30 need_worker=200
  [ "$WEIGHTS_MODE" = local ] && need_head=200
  [ "$SKIP_DOWNLOAD" = 1 ] && { need_head=30; need_worker=30; }
  local free_head free_worker
  free_head=$(disk_free_g "$HOME"); free_worker=$(WSSH "df -BG --output=avail \$HOME | tail -1 | tr -dc 0-9")
  [ "${free_head:-0}" -ge "$need_head" ] || die "head needs ~${need_head}G free in \$HOME (has ${free_head:-?}G)"
  [ "${free_worker:-0}" -ge "$need_worker" ] || die "worker needs ~${need_worker}G free in \$HOME (has ${free_worker:-?}G)"
  ok "hosts verified"
}

# ---------- §2 image ---------------------------------------------------------
pull_image() {
  if [ "$SKIP_PULL" = 1 ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
    ok "image present (pull skipped): $IMAGE"
  else
    log "pulling $IMAGE (~25 GiB on first pull)"
    docker pull "$IMAGE" || die "docker pull failed — check network/GHCR status"
  fi
  local head_id worker_id
  head_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
  worker_id=$(WSSH "docker image inspect --format '{{.Id}}' '$IMAGE' 2>/dev/null" || true)
  if [ "$head_id" != "$worker_id" ]; then
    log "shipping image to worker (docker save | ssh docker load, ~25 GiB)"
    docker save "$IMAGE" | WSSH docker load || die "image ship to worker failed"
  fi
  ok "image on both boxes: $IMAGE"
}

# ---------- §3 weights -------------------------------------------------------
dl_in_container() { # dl_in_container <runner...> -- <hf_repo> <host_dir>
  local run=() a
  while [ "$1" != "--" ]; do run+=("$1"); shift; done; shift
  local repo=$1 dir=$2
  "${run[@]}" "mkdir -p '$dir' && docker run --rm -v '$dir:/dl' -v '$dir/.hf:/root/.cache/huggingface' --entrypoint python3 '$IMAGE' -c \"from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='/dl')\""
}
have_exl3() { # have_exl3 <runner...> -- <dir>
  local run=() ; while [ "$1" != "--" ]; do run+=("$1"); shift; done; shift
  "${run[@]}" "test -f '$1/config.json' && [ \$(ls '$1'/*.safetensors 2>/dev/null | wc -l) -ge 120 ]"
}

download_models() {
  [ "$SKIP_DOWNLOAD" = 1 ] && { log "downloads skipped"; return; }
  echo
  log "DFlash2 drafter ($DFLASH_REPO) is CC BY-NC-ND 4.0 (research/eval)."
  log "It is downloaded from its source repository and never redistributed."
  echo
  # drafter: needed on BOTH boxes (each TP rank loads it)
  WSSH "test -f '$DFLASH_DIR/config.json'" || dl_in_container WSSH -- "$DFLASH_REPO" "$DFLASH_DIR"
  test -f "$DFLASH_DIR/config.json" || dl_in_container bash -c -- "$DFLASH_REPO" "$DFLASH_DIR"

  # EXL3 weights per topology
  if [ "$WEIGHTS_MODE" = local ]; then
    if ! have_exl3 bash -c -- "$MODEL_HOST_PATH"; then
      log "downloading EXL3 weights to head:$MODEL_HOST_PATH (~176 GiB)"
      dl_in_container bash -c -- "$EXL3_REPO" "$MODEL_HOST_PATH" \
        || dl_in_container bash -c -- "$EXL3_REPO_FALLBACK" "$MODEL_HOST_PATH" \
        || die "EXL3 download failed from both $EXL3_REPO and $EXL3_REPO_FALLBACK"
    fi
    if ! have_exl3 WSSH -- "$MODEL_HOST_PATH"; then
      log "rsyncing weights head -> worker over the rail ($WORKER_RAIL_IP)"
      rsync -a --info=progress2 -e "ssh -o BatchMode=yes" "$MODEL_HOST_PATH/" "${WORKER_USER:+$WORKER_USER@}$WORKER_RAIL_IP:$MODEL_HOST_PATH/" \
        || rsync -a --info=progress2 -e "ssh -o BatchMode=yes" "$MODEL_HOST_PATH/" "$WORKER:$MODEL_HOST_PATH/" \
        || die "rsync to worker failed"
    fi
  else
    if ! have_exl3 WSSH -- "$MODEL_HOST_PATH"; then
      log "downloading EXL3 weights to worker:$MODEL_HOST_PATH (~176 GiB)"
      dl_in_container WSSH -- "$EXL3_REPO" "$MODEL_HOST_PATH" \
        || dl_in_container WSSH -- "$EXL3_REPO_FALLBACK" "$MODEL_HOST_PATH" \
        || die "EXL3 download failed from both $EXL3_REPO and $EXL3_REPO_FALLBACK"
    fi
    log "ensuring containerized NFS export on the worker (port $NFS_PORT)"
    WSSH "docker ps --format '{{.Names}}' | grep -q '^nfs-exl3\$'" || WSSH "docker rm -f nfs-exl3 2>/dev/null; docker run -d --name nfs-exl3 --restart unless-stopped --privileged -p $NFS_PORT:2049 -v '$MODEL_HOST_PATH:/export/glm53-exl3:ro' -v /lib/modules:/lib/modules:ro -e NFS_EXPORT_0='/export/glm53-exl3 *(ro,no_subtree_check,fsid=0,insecure)' erichough/nfs-server" \
      || die "NFS export container failed on the worker"
  fi
  ok "weights in place ($WEIGHTS_MODE mode)"
}

# ---------- §4 per-box scripts + serve config --------------------------------
write_env() { # write_env <target: head|worker>
  local nccl_if nccl_hca out
  if [ "$1" = head ]; then nccl_if=$HEAD_NCCL_IF; nccl_hca=$HEAD_NCCL_HCA; else nccl_if=$WORKER_NCCL_IF; nccl_hca=$WORKER_NCCL_HCA; fi
  out=$(cat <<EOF
# Written by install.sh $(date -u +%Y-%m-%dT%H:%MZ) — command-line env wins.
: "\${HEAD_RAIL_IP:=$HEAD_RAIL_IP}"
: "\${WORKER_RAIL_IP:=$WORKER_RAIL_IP}"
: "\${NCCL_IF:=$nccl_if}"
: "\${NCCL_HCA:=$nccl_hca}"
: "\${NCCL_SUBNET:=$NCCL_SUBNET}"
: "\${WEIGHTS_MODE:=$WEIGHTS_MODE}"
: "\${MODEL_HOST_PATH:=$MODEL_HOST_PATH}"
: "\${DFLASH_DIR:=$DFLASH_DIR}"
: "\${IMAGE:=$IMAGE}"
: "\${PORT:=$PORT}"
: "\${NFS_PORT:=$NFS_PORT}"
: "\${MPORT:=$MPORT}"
: "\${MAX_LEN:=$MAX_LEN}"
: "\${SPEC:=$SPEC}"
: "\${MTP:=$MTP}"
: "\${DFLASH_TOKENS:=$DFLASH_TOKENS}"
: "\${EAGER:=$EAGER}"
: "\${SKIP_MM_PROFILING:=$SKIP_MM_PROFILING}"
: "\${BLOCK_SIZE:=$BLOCK_SIZE}"
: "\${KV_DTYPE:=$KV_DTYPE}"
: "\${KV_CACHE_MEMORY:=$KV_CACHE_MEMORY}"
: "\${KV_SKIP_LAYERS:=$KV_SKIP_LAYERS}"
: "\${ATTN_BACKEND:=$ATTN_BACKEND}"
: "\${MNBT:=$MNBT}"
: "\${GMU:=$GMU}"
: "\${MAX_SEQS:=$MAX_SEQS}"
: "\${MM_CACHE_GB:=$MM_CACHE_GB}"
EOF
)
  if [ "$1" = head ]; then printf '%s\n' "$out" > "$HOME/.glm53-serve.env"
  else printf '%s\n' "$out" | WSSH "cat > \$HOME/.glm53-serve.env"; fi
}

install_scripts() {
  write_env head; write_env worker
  install -m 0755 "$SCRIPT_DIR/scripts/launch-glm53-vllm-tp2.sh" "$HOME/launch-glm53-vllm-tp2.sh"
  install -m 0755 "$SCRIPT_DIR/scripts/glm53-warmup.sh" "$HOME/glm53-warmup.sh"
  scp -q -o BatchMode=yes "$SCRIPT_DIR/scripts/launch-glm53-vllm-tp2.sh" "$WORKER:launch-glm53-vllm-tp2.sh"
  WSSH "chmod +x \$HOME/launch-glm53-vllm-tp2.sh"
  ok "launch scripts + serve config installed on both boxes"
}

# ---------- §5 launch --------------------------------------------------------
start_server() {
  [ "$NO_START" = 1 ] && { log "--no-start: skipping launch. Worker: ./launch-glm53-vllm-tp2.sh 1, then head: ./launch-glm53-vllm-tp2.sh 0"; return; }
  # Tear down a live head FIRST: a fresh worker otherwise rendezvouses with
  # the OLD head's TCP store and dies of connection-reset the moment that
  # head is replaced (leaving the new head waiting on a dead worker).
  docker rm -f vllm_glm53 >/dev/null 2>&1 || true
  log "launching worker (rank 1)"
  WSSH "\$HOME/launch-glm53-vllm-tp2.sh 1" || die "worker launch failed"
  sleep 30
  log "launching head (rank 0); weight load + engine init takes ~13 min"
  "$HOME/launch-glm53-vllm-tp2.sh" 0 || die "head launch failed"
  local t0 elapsed
  t0=$(date +%s)
  until curl -sf -m5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
    sleep 15
    docker ps --format '{{.Names}}' | grep -q '^vllm_glm53$' || {
      echo; docker logs vllm_glm53 2>&1 | tail -30
      die "head container exited during boot (last log lines above). If it wedged at ~90% of shard load in local mode, re-run with --nfs."
    }
    elapsed=$(( $(date +%s) - t0 ))
    [ "$elapsed" -gt 1800 ] && die "API not up after 30 min — docker logs vllm_glm53"
  done
  ok "API up after $(( $(date +%s) - t0 ))s"
  log "warming JIT shapes (~20 s)"
  API_BASE="http://localhost:$PORT" bash "$HOME/glm53-warmup.sh" || warn "warmup had failures (serving continues)"
  local smoke
  smoke=$(curl -s -m 90 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}],"temperature":0,"max_tokens":200}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
  case "$smoke" in *391*) ok "smoke test passed: '$smoke'" ;; *) die "smoke test failed (got: '$smoke')" ;; esac
}

verify_hosts
pull_image
download_models
install_scripts
start_server

echo
if [ "$NO_START" = 1 ]; then
  ok "Prepared (not launched)."
  exit 0
fi
ok "Done. API: http://localhost:$PORT/v1  (model id: glm-5.3-flash)"
echo "  logs:   docker logs -f vllm_glm53"
echo "  stop:   docker rm -f vllm_glm53   (head)  |  ssh $WORKER docker rm -f vllm_glm53"
echo "  bench:  python3 scripts/bench_decode_miaai.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence"
echo "  fp8 KV option (770k-token pool):  KV_DTYPE=fp8_e4m3 on both launches"
