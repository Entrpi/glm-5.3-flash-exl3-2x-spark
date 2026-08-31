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
#     nfs-exl3 + the worker-built glm53-nfs-server:kit image in --nfs mode,
#     exl3weights volume)
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
IMAGE="${IMAGE:-ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2-glmnext}"
NFS_PORT="${NFS_PORT:-12049}"
# --nfs mode export server image. Default: the kit's own minimal NFSv4-only
# image (nfs/Dockerfile), built natively ON the worker at install time —
# published NFS server images are amd64-only and crash-loop on the aarch64
# Spark unless qemu binfmt happens to be registered (issue #1). Override to
# run a prebuilt image instead (skips the build).
NFS_IMAGE="${NFS_IMAGE:-glm53-nfs-server:kit}"
PORT="${PORT:-8000}"

# Serving knobs (per-knob rationale in scripts/launch-glm53-vllm-tp2.sh).
# write_env forwards a knob into ~/.glm53-serve.env ONLY when the user set it
# (.env or environment) — unset knobs stay out of the file, so the launcher's
# validated defaults remain the single source of truth across kit upgrades.
SERVE_KNOBS=(MPORT MAX_LEN SPEC MTP DFLASH_TOKENS EAGER SKIP_MM_PROFILING
  BLOCK_SIZE KV_DTYPE KV_CACHE_MEMORY KV_SKIP_LAYERS ATTN_BACKEND MNBT GMU
  MAX_SEQS MM_CACHE_GB LOAD_FORMAT CACHE_HOST_PATH VLLM_EXL3_STANDARD_FUSED
  VLLM_NVFP4_MLA_DYNAMIC_SCALE VLLM_DISABLED_KERNELS GLM53_TOPK_FIX_SO)

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

# Swap surgery: 32G swapfile + fstab persistence. Deliberately quote-free so
# it survives the WSSH "sudo -n bash -c '...'" nesting.
SWAP_FIX_CMD='test -f /swap-glm53 || (fallocate -l 32G /swap-glm53 && chmod 600 /swap-glm53 && mkswap /swap-glm53); swapon /swap-glm53 2>/dev/null; grep -q /swap-glm53 /etc/fstab || echo /swap-glm53 none swap sw 0 0 >> /etc/fstab'
grow_swap() {
  if [ "${SWAP_HEAD:-0}" -lt 30 ]; then
    log "growing swap on the head (sudo may prompt for your password)"
    sudo bash -c "$SWAP_FIX_CMD" \
      && ok "head SwapTotal now $(awk '/^SwapTotal/{print int($2/1048576)}' /proc/meminfo)G" \
      || warn "head swap surgery failed — run manually: sudo bash -c '$SWAP_FIX_CMD'"
  fi
  if [ "${SWAP_WORKER:-0}" -lt 30 ]; then
    log "growing swap on the worker (needs passwordless sudo over SSH)"
    if WSSH "sudo -n bash -c '$SWAP_FIX_CMD'" >/dev/null 2>&1; then
      ok "worker SwapTotal now $(WSSH "awk '/^SwapTotal/{print int(\$2/1048576)}' /proc/meminfo")G"
    else
      warn "worker swap surgery failed (no passwordless sudo?) — run ON THE WORKER: sudo bash -c '$SWAP_FIX_CMD'"
    fi
  fi
}

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
  # Swap prerequisite (measured 2026-08-31): weight load consumes ALL
  # available swap on the head in BOTH weights modes (reference pair: full
  # 32 GiB used, 0.8 GiB MemFree floor; even the worker pegs its 16 GiB).
  # Stock 16 GiB OOMs the head at ~90% of shard load (NV_ERR_NO_MEMORY).
  SWAP_HEAD=$(awk '/^SwapTotal/{print int($2/1048576)}' /proc/meminfo)
  SWAP_WORKER=$(WSSH "awk '/^SwapTotal/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null || echo 0)
  local swap_short=""
  [ "${SWAP_HEAD:-0}" -ge 30 ] || swap_short="head(${SWAP_HEAD}G)"
  [ "${SWAP_WORKER:-0}" -ge 30 ] || swap_short="$swap_short worker(${SWAP_WORKER}G)"
  # ${VAR-default} mirrors the launcher: UNSET tracks the launcher's
  # instanttensor default (no swap dependency); an explicitly EMPTY
  # LOAD_FORMAT= opts into the page-cached loader and needs the swap.
  local effective_load_format="${LOAD_FORMAT-instanttensor}"
  if [ -n "$swap_short" ] && [ "$effective_load_format" = "instanttensor" ]; then
    warn "swap below 32G ($swap_short) — fine with the default direct-I/O loader, but grow it before opting into LOAD_FORMAT= (page-cached)"
  fi
  if [ -n "$swap_short" ] && [ "$effective_load_format" != "instanttensor" ]; then
    warn "INSUFFICIENT SWAP:$swap_short — need >=32G per box."
    warn "Weight load consumes ALL available swap (measured: the reference pair's"
    warn "full 32 GiB, in both local and --nfs modes). With stock 16 GiB the head"
    warn "dies deterministically at ~90% of shard load (NV_ERR_NO_MEMORY)."
    warn "Alternative: LOAD_FORMAT=instanttensor in .env bypasses the page cache."
    if [ -t 0 ]; then
      read -r -p "Grow swap now: 32G /swap-glm53 via sudo ([y]es / [c]ontinue anyway / [A]bort)? " ans
      case "$ans" in
        y|Y) grow_swap ;;
        c|C) warn "continuing with insufficient swap — expect the head to die during weight load" ;;
        *) die "aborted — grow swap or set LOAD_FORMAT=instanttensor, then re-run" ;;
      esac
    elif [ "$FORCE" = 1 ]; then
      warn "continuing (--force) with insufficient swap — expect the head to die during weight load"
    else
      die "insufficient swap (details above). Grow swap, set LOAD_FORMAT=instanttensor in .env, or re-run with --force"
    fi
  fi
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
  # Compare content digests (RepoDigests), not store-local .Id: the
  # containerd image store assigns multi-arch images different .Ids for
  # identical content, which forced pointless ~25 GiB ships.
  local head_dig worker_dig head_id worker_id
  head_dig=$(docker image inspect --format '{{join .RepoDigests ","}}' "$IMAGE" 2>/dev/null || true)
  worker_dig=$(WSSH "docker image inspect --format '{{join .RepoDigests \",\"}}' '$IMAGE' 2>/dev/null" || true)
  if [ -n "$head_dig" ] && [ "$head_dig" = "$worker_dig" ]; then
    ok "image on both boxes: $IMAGE"
    return
  fi
  if [ -z "$head_dig" ]; then
    # Locally built image (no registry digest): fall back to store .Id.
    head_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
    worker_id=$(WSSH "docker image inspect --format '{{.Id}}' '$IMAGE' 2>/dev/null" || true)
    if [ "$head_id" = "$worker_id" ]; then
      ok "image on both boxes: $IMAGE"
      return
    fi
  fi
  # Worker pulls directly first: docker save|load of a containerd-store
  # multi-arch OCI image can fail or ship the wrong platform (issue #8);
  # a registry pull always resolves the right one.
  log "image missing/stale on worker — trying a direct pull there"
  if WSSH "docker pull '$IMAGE'" >/dev/null 2>&1; then
    ok "image on both boxes: $IMAGE (worker pulled directly)"
    return
  fi
  log "worker pull failed; shipping image (docker save | ssh docker load, ~25 GiB)"
  docker save "$IMAGE" | WSSH docker load || die "image ship to worker failed"
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
    if ! WSSH "docker ps --format '{{.Names}}' | grep -q '^nfs-exl3\$'"; then
      if [ "$NFS_IMAGE" = "glm53-nfs-server:kit" ]; then
        log "building the kit's NFSv4 server image on the worker (native arch)"
        tar -c -C "$SCRIPT_DIR/nfs" . | WSSH "docker build -q -t '$NFS_IMAGE' -" >/dev/null \
          || die "NFS server image build failed on the worker"
      fi
      WSSH "docker rm -f nfs-exl3 2>/dev/null; docker run -d --name nfs-exl3 --restart unless-stopped --privileged -p $NFS_PORT:2049 -v '$MODEL_HOST_PATH:/export/glm53-exl3:ro' -v /lib/modules:/lib/modules:ro -e NFS_EXPORT_0='/export/glm53-exl3 *(ro,no_subtree_check,fsid=0,insecure)' '$NFS_IMAGE'" \
        || die "NFS export container failed on the worker"
      sleep 3
      WSSH "docker ps --format '{{.Names}}' | grep -q '^nfs-exl3\$'" || {
        WSSH "docker logs nfs-exl3 2>&1 | tail -8" || true
        die "NFS export container is not running (last logs above; amd64-only image on aarch64? see issue #1)"
      }
    fi
  fi
  ok "weights in place ($WEIGHTS_MODE mode)"
}

# ---------- §4 per-box scripts + serve config --------------------------------
write_env() { # write_env <target: head|worker>
  local nccl_if nccl_hca out k
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
EOF
)
  # User-set serving knobs only (see SERVE_KNOBS). No-colon `=` (not `:=`):
  # a deliberately EMPTY value (e.g. KV_DTYPE= -> bf16) survives the write,
  # and launch-time env still overrides either way.
  for k in "${SERVE_KNOBS[@]}"; do
    [ -n "${!k+x}" ] && out="$out
: \"\${$k=${!k}}\""
  done
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
  local t0 elapsed worker_strikes=0
  t0=$(date +%s)
  until curl -sf -m5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
    sleep 15
    docker ps --format '{{.Names}}' | grep -q '^vllm_glm53$' || {
      echo; docker logs vllm_glm53 2>&1 | tail -30
      die "head container exited during boot (last log lines above). If it wedged or OOMed at ~90% of shard load: grow swap to >=32 GiB on both boxes (stock 16 GiB is not enough — see README), or set LOAD_FORMAT=instanttensor in .env, then re-run."
    }
    # A dead worker otherwise burns the full 30-min timeout while the head
    # waits on rendezvous. 3 consecutive strikes so one flaky ssh probe
    # can't kill a healthy boot.
    if WSSH "docker ps --format '{{.Names}}' | grep -q '^vllm_glm53\$'" >/dev/null 2>&1; then
      worker_strikes=0
    else
      worker_strikes=$(( worker_strikes + 1 ))
      if [ "$worker_strikes" -ge 3 ]; then
        echo; WSSH "docker logs vllm_glm53 2>&1 | tail -30" || true
        die "worker container is gone during boot (last worker log lines above, if any). Fix the worker, then re-run."
      fi
    fi
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
