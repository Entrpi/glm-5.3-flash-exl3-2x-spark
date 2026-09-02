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
#   3. download weights (EXL3 ~176 GiB; DFlash2 drafter ~1.3 GiB) per the
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
#   local-inference-lab GLM-5.3-Flash-DFlash2-MXFP8 (the drafter's 8-bit copy, default)
#   vLLM + local-inference-lab fork lineage - eugr's spark-vllm-docker build
#   turboderp exllamav3 - tpurtell sparkinfer-glmrt (b12x) - FlashInfer
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- §0 config + flags ------------------------------------------------
case " $* " in *" --version "*|*" --check-update "*|*" --help "*|*" -h "*) ;;   # read-only flags never create .env
  *) [ -f "$SCRIPT_DIR/.env" ] || { [ -f "$SCRIPT_DIR/.env.example" ] && cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env" && echo "NOTE: created .env from .env.example — edit it and re-run if these defaults are wrong."; } ;;
esac
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
DFLASH_DIR="${DFLASH_DIR:-$HOME/models/glm53-dflash2-mxfp8}"
EXL3_REPO="${EXL3_REPO:-brandonmusic/GLM-5.3-Flash-tr3-4bpw}"
EXL3_REPO_FALLBACK="${EXL3_REPO_FALLBACK:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}"
DFLASH_REPO="${DFLASH_REPO:-local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8}"
# Revision of the drafter repo to fetch. The default pins the validated commit
# of the default repo; any other DFLASH_REPO fetches its latest revision.
if [ -z "${DFLASH_REVISION+x}" ]; then
  case "$DFLASH_REPO" in
    local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8) DFLASH_REVISION=62f758c0a0e19b9cb76fc098c911b8ed76daff5b ;;
    *) DFLASH_REVISION="" ;;
  esac
fi
DEFAULT_IMAGE=ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
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
  VLLM_NVFP4_MLA_DYNAMIC_SCALE VLLM_DISABLED_KERNELS GLM53_TOPK_FIX_SO
  MIXED_PREFILL_DECODE_WEIGHT MIXED_PREFILL_CAP PREFIX_MATCH_UNIT
  VLLM_USE_B12X_FP8_GEMM VLLM_B12X_MXFP8_MAX_M VLLM_DFLASH_FP8_DRAFT_HEAD NCCL_NCHANNELS
  MEM_USED_MAX_GB GLM53_NO_UPDATE_CHECK)
KIT_VERSION=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo unknown)
GLM53_UPDATE_URL="${GLM53_UPDATE_URL:-https://raw.githubusercontent.com/Entrpi/glm-5.3-flash-exl3-2x-spark/main/LATEST}"

SKIP_PULL=0 SKIP_DOWNLOAD=0 NO_START=0 FORCE=0 PRUNE_OLD=0 YES=0 ROLLBACK=0 NO_ROLLBACK=0 REMOVE_OLD=0 KEEP_OLD=0 CLEANUP=0
usage() {
  cat <<'EOF'
Usage: ./install.sh [--nfs|--local-both] [--skip-pull] [--skip-download]
                    [--no-start] [--yes] [--remove-old|--keep-old|--prune-old-images]
                    [--no-rollback] [--force] [--rollback] [--cleanup] [--version] [--check-update] [--help]

  --nfs           weights only on the worker, NFS-exported to the head
                  (the reference kit's validated-production topology)
  --local-both    weights on both boxes (default; simplest, ~176 GiB/box.
                  If the head swap-wedges at ~90% of load, re-run --nfs)
  --skip-pull     keep the local image (no GHCR pull)
  --skip-download weights already present on the right boxes
  --no-start      prepare everything but do not launch
  --remove-old    after the new release passes its smoke test, delete the leftovers of
                  earlier releases on both boxes without asking: superseded kit image tags
                  (~18 GiB each) and the old bf16 drafter directory (2.3 GiB)
  --keep-old      keep them without asking (no flag: ask when a terminal is attached,
                  otherwise keep and say so)
  --prune-old-images  images only, no prompt (older spelling of --remove-old)
  --force         downgrade hardware-check and memory-preflight failures to warnings
                  (MEM_USED_MAX_GB=<gb> in .env changes the 6 GB preflight limit; 0 skips it)
  --yes           accept the upgrade summary without asking
  --no-rollback   if the new release fails to start, leave its containers running for
                  debugging instead of restoring the previous release
  --rollback      restore the previous release (files saved by the last upgrade) and relaunch
  --cleanup       only list the leftovers of earlier releases and offer to remove them
                  (nothing is relaunched; combine with --remove-old / --keep-old)
  --version       print this kit's release and what the boxes have installed
  --check-update  compare this checkout with the published release list (LATEST)

An upgrade shows what it is about to change (image, weights, serve config
diff, leftovers, .env notes) and asks before touching anything. The previous
release's files are kept as *.prev on both boxes; if the new release does not
come up or fails its smoke test, it is restored automatically.

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
    --prune-old-images) PRUNE_OLD=1 ;;
    --yes|-y) YES=1 ;;
    --no-rollback) NO_ROLLBACK=1 ;;
    --rollback) ROLLBACK=1 ;;
    --cleanup) CLEANUP=1 ;;
    --remove-old) REMOVE_OLD=1 ;;
    --keep-old) KEEP_OLD=1 ;;
    --version) echo "kit $KIT_VERSION ($(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"; [ -f "$HOME/.glm53-kit-release" ] && sed 's/^/  installed: /' "$HOME/.glm53-kit-release" | grep -E "KIT_(IMAGE|DATE|COMMIT)"; exit 0 ;;
    --check-update)
      latest=$(curl -fsS --max-time 5 "$GLM53_UPDATE_URL" 2>/dev/null | head -1 | tr -d '\r')
      echo "kit $KIT_VERSION"
      [ -n "$latest" ] || { echo "update check failed (network?)"; exit 1; }
      key() { local v="${1#v}"; v="${v%%-*}"; printf '%s' "$v" | awk -F. '{printf "%d.%03d.%03d", $1, $2, $3}'; }
      if [ "$(key "$latest")" \> "$(key "$KIT_VERSION")" ]; then echo "update available: $latest — git pull in this checkout, then ./install.sh"; else echo "latest is $latest — up to date"; fi
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $arg"; usage; exit 2 ;;
  esac
done
[ "$REMOVE_OLD" = 1 ] && [ "$KEEP_OLD" = 1 ] && { echo "pass at most one of --remove-old / --keep-old"; exit 2; }

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }
# ask_tty <prompt> -> 0 = yes, 1 = no, 2 = no terminal. Reads /dev/tty so a
# piped run (curl | bash, nohup) still asks when a terminal is attached.
ask_tty() {
  local ans=""
  if ( : < /dev/tty ) 2>/dev/null; then
    printf '%s' "$1" > /dev/tty; read -r ans < /dev/tty || true
    case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
  fi
  return 2
}

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

# ---------- §0b .env lint + installed release --------------------------------
# .env is sourced wholesale and only SERVE_KNOBS reach the boxes, so a typo or
# a retired key is otherwise silently ignored. Notes are collected here and
# shown in the upgrade summary.
ENV_NOTES=()
lint_env() {
  local envf="$SCRIPT_DIR/.env" known key val line
  [ -f "$envf" ] || return 0
  known=" $(grep -oE '^#? *[A-Z_][A-Z0-9_]*=' "$SCRIPT_DIR/.env.example" 2>/dev/null | tr -d '# =' | sort -u | tr '\n' ' ') ${SERVE_KNOBS[*]} "
  while IFS= read -r line || [ -n "$line" ]; do
    key=$(printf '%s' "$line" | sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Z_][A-Z0-9_]*)=.*/\2/p')
    [ -n "$key" ] || continue
    val=${line#*=}; val=${val%%#*}
    val=$(printf '%s' "$val" | sed -E 's/^[[:space:]]*"?//; s/"?[[:space:]]*$//')
    case "$key" in
      IMAGE)
        if [ -n "$val" ] && [ "$val" != "$DEFAULT_IMAGE" ]; then
          ENV_NOTES+=("IMAGE is pinned to ${val##*:}; this kit release is ${DEFAULT_IMAGE##*:}. Remove the IMAGE line from .env to upgrade (v1/v2 tags are refused by the launcher with fine-grained settings).")
        fi ;;
      GLM53_TOPK_FIX_SO) ENV_NOTES+=("GLM53_TOPK_FIX_SO is a v1-era override; the fix is baked into every image since v2 and the launcher ignores it. Remove the line.") ;;
      KV_CACHE_MEMORY) [ "$val" = 12400000000 ] && ENV_NOTES+=("KV_CACHE_MEMORY=12400000000 is the v2.1 advice; v2.2 and later default to 14.4 GB (1.29M-token pool). Remove the line unless memory is short.") ;;
      MIXED_PREFILL_CAP) [ "$val" = "-1" ] && ENV_NOTES+=("MIXED_PREFILL_CAP=-1 selects the legacy v2 skip mode; the v2.2+ fair scheduler default is 512.") ;;
    esac
    case "$known" in *" $key "*) ;; *) ENV_NOTES+=("unknown setting $key in .env: not used by this release (check the spelling against .env.example).") ;; esac
  done < "$envf"
  # drafter directory vs repo format (MXFP8 copies carry hf_quant_config.json)
  if [ -f "$DFLASH_DIR/config.json" ] && [ ! -f "$DFLASH_DIR/hf_quant_config.json" ]; then
    case "$DFLASH_REPO" in
      *MXFP8*) ENV_NOTES+=("DFLASH_DIR=$DFLASH_DIR holds the bf16 drafter while DFLASH_REPO is the MXFP8 copy. Keep bf16 with DFLASH_REPO=incoai/GLM-5.3-Flash-DFlash2, or unset DFLASH_DIR to use the MXFP8 default directory.") ;;
    esac
  fi
}

# The release stamp (~/.glm53-kit-release on both boxes) records what an
# install left behind; pre-2.3 installs have none, so fall back to the image
# pinned in the serve config or the running container.
STAMP="$HOME/.glm53-kit-release"
PREV_IMAGE="" PREV_DESC="" UPGRADE=0
detect_installed() {
  if [ -f "$STAMP" ]; then
    PREV_IMAGE=$(sed -n 's/^KIT_IMAGE=//p' "$STAMP" | head -1)
    PREV_DESC="${PREV_IMAGE##*:} (installed $(sed -n 's/^KIT_DATE=//p' "$STAMP" | head -1), kit $(sed -n 's/^KIT_COMMIT=//p' "$STAMP" | head -1))"
  elif [ -f "$HOME/.glm53-serve.env" ]; then
    PREV_IMAGE=$(sed -nE 's/^: "\$\{IMAGE:=([^}]*)\}"$/\1/p' "$HOME/.glm53-serve.env" | head -1)
    [ -n "$PREV_IMAGE" ] || PREV_IMAGE=$(docker inspect vllm_glm53 --format '{{.Config.Image}}' 2>/dev/null || true)
    PREV_DESC="${PREV_IMAGE:+${PREV_IMAGE##*:} }(earlier kit release without a stamp; from ~/.glm53-serve.env)"
  fi
  [ -f "$HOME/.glm53-serve.env" ] && UPGRADE=1
  return 0
}

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
MEM_USED_MAX_GB="${MEM_USED_MAX_GB:-6}"
# Hotfix overlay dirs (~/glm53-hotfix, -fi, -b12x) were debugging aids for the
# v1-dflash2 image; the launcher binds them over whatever image runs, so an
# old install's overlays would shadow the current image's code. Refuse to
# upgrade over them (--force downgrades to a warning).
overlay_check() { # overlay_check <where> <runner...>
  local where=$1; shift
  local run=("$@") found
  found=$("${run[@]}" 'for d in $HOME/glm53-hotfix $HOME/glm53-hotfix-fi $HOME/glm53-hotfix-b12x; do [ -d "$d" ] && echo "$d"; done' 2>/dev/null | tr '\n' ' ')
  [ -z "$found" ] && return 0
  warn "$where: hotfix overlay dir(s) present: $found"
  warn "$where: they would be bound over $IMAGE. Move them aside: for d in $found; do mv \"\$d\" \"\$d.retired-\$(date +%F)\"; done"
  if [ "$FORCE" = 1 ]; then warn "$where: continuing under --force (overlays stay active)"; else die "$where: remove or move the overlay dirs, then re-run (or --force to keep them)"; fi
}
mem_check() { # mem_check <where> <runner...>  — system memory in use before launch
  local where=$1; shift
  local run=("$@")
  [ "$MEM_USED_MAX_GB" = 0 ] && return 0
  if "${run[@]}" 'test -n "$(docker ps -q -f name=vllm_glm53 2>/dev/null)"'; then
    log "$where: vllm_glm53 is running — memory preflight deferred to the launcher"
    return 0
  fi
  local used_mb
  used_mb=$("${run[@]}" "free -m | awk 'NR==2{print \$3}'" 2>/dev/null || echo 0)   # free(1) "used" column
  if [ "${used_mb:-0}" -gt $((MEM_USED_MAX_GB * 1024)) ]; then
    warn "$where: $((used_mb/1024)) GB of system memory in use before launch (limit ${MEM_USED_MAX_GB} GB); top consumers:"
    "${run[@]}" "ps -eo rss,comm --sort=-rss | awk 'NR>1 && NR<=6 {printf \"    %6d MB  %s\n\", \$1/1024, \$2}'" 2>/dev/null
    if [ "$FORCE" = 1 ]; then
      warn "$where: continuing under --force; consider a smaller KV_CACHE_MEMORY (~90k pool tokens per GB)"
    else
      die "$where: stop the other consumers, set a smaller KV_CACHE_MEMORY in .env, or re-run with MEM_USED_MAX_GB=<gb> (0 skips) / --force"
    fi
  else
    ok "$where: $((used_mb/1024)) GB of system memory in use (limit ${MEM_USED_MAX_GB} GB)"
  fi
}

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
  # Disk: ~200 GiB only where the EXL3 weights still have to land; a box
  # that already holds them (any earlier release) needs ~30 GiB for the image.
  local need_head=30 need_worker=30
  if [ "$SKIP_DOWNLOAD" != 1 ]; then
    if [ "$WEIGHTS_MODE" = local ]; then
      have_exl3 bash -c -- "$MODEL_HOST_PATH" || need_head=200
      have_exl3 WSSH -- "$MODEL_HOST_PATH" || need_worker=200
    else
      have_exl3 WSSH -- "$MODEL_HOST_PATH" || need_worker=200
    fi
  fi
  local free_head free_worker
  free_head=$(disk_free_g "$HOME"); free_worker=$(WSSH "df -BG --output=avail \$HOME | tail -1 | tr -dc 0-9")
  [ "${free_head:-0}" -ge "$need_head" ] || die "head needs ~${need_head}G free in \$HOME (has ${free_head:-?}G)"
  [ "${free_worker:-0}" -ge "$need_worker" ] || die "worker needs ~${need_worker}G free in \$HOME (has ${free_worker:-?}G)"
  # Memory preflight (same rule as the launcher): the validated defaults
  # assume a lightly loaded headless box with < MEM_USED_MAX_GB (6) of
  # system memory in use before launch. A box already running vllm_glm53 is
  # skipped here (the launcher re-checks after removing it). --force
  # downgrades a failure to a warning; MEM_USED_MAX_GB=0 skips the check.
  mem_check head bash -c
  mem_check worker WSSH
  overlay_check head bash -c
  overlay_check worker WSSH
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


# Leftovers of earlier releases: superseded kit image tags and the bf16
# drafter directory. Listed in the summary; removal is offered only AFTER the
# new release has passed its smoke test (ds4-on-spark pattern), and only with
# consent: --remove-old / --keep-old, a terminal prompt, or kept by default.
OLD_DRAFTER_DIR="$HOME/models/glm53-dflash2"
list_old_images() { # list_old_images <runner...>
  local repo="${IMAGE%%:*}"
  "$@" "docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep '^$repo:' | grep -v '^$IMAGE ' | grep -v ':latest '" 2>/dev/null
}
# Directories written by earlier installers' downloads (or the serving
# container) contain root-owned files; a plain rm fails without sudo, so fall
# back to emptying them from a throwaway root container.
rm_dir_any_owner() { # rm_dir_any_owner <runner...>   (removes $OLD_DRAFTER_DIR)
  "$@" "[ -d '$OLD_DRAFTER_DIR' ] || exit 0; rm -rf -- '$OLD_DRAFTER_DIR' 2>/dev/null && exit 0; docker run --rm -v '$OLD_DRAFTER_DIR:/rm' --entrypoint sh '$IMAGE' -c 'rm -rf /rm/* /rm/.[!.]* 2>/dev/null; true' >/dev/null 2>&1; rmdir -- '$OLD_DRAFTER_DIR'"
}
offer_cleanup() {
  local head_imgs worker_imgs dfl="" n=0 what=""
  head_imgs=$(list_old_images bash -c); worker_imgs=$(list_old_images WSSH)
  [ -n "$head_imgs" ] && { what="$what superseded images (head): $(echo "$head_imgs" | awk '{print $1}' | sed 's/.*://' | tr '\n' ' ')"; n=1; }
  [ -n "$worker_imgs" ] && { what="$what superseded images (worker): $(echo "$worker_imgs" | awk '{print $1}' | sed 's/.*://' | tr '\n' ' ')"; n=1; }
  if [ "$DFLASH_DIR" != "$OLD_DRAFTER_DIR" ] && { [ -d "$OLD_DRAFTER_DIR" ] || WSSH "test -d '$OLD_DRAFTER_DIR'"; }; then
    dfl="$OLD_DRAFTER_DIR (bf16 drafter, 2.3 GiB per box)"; what="$what $dfl"; n=1
  fi
  [ "$n" = 1 ] || return 0
  log "leftovers of earlier releases:"
  [ -n "$head_imgs" ] && echo "$head_imgs" | sed 's/^/    head:   /'
  [ -n "$worker_imgs" ] && echo "$worker_imgs" | sed 's/^/    worker: /'
  [ -n "$dfl" ] && echo "    both:   $dfl"
  local do_rm=0 do_imgs=1 do_dfl=1
  if [ "$REMOVE_OLD" = 1 ]; then do_rm=1
  elif [ "$PRUNE_OLD" = 1 ]; then do_rm=1; do_dfl=0
  elif [ "$KEEP_OLD" = 1 ]; then do_rm=0
  else
    ask_tty "Remove them? (${IMAGE##*:} is the release in use) [y/N] "
    case $? in 0) do_rm=1 ;; 1) do_rm=0 ;; 2) log "no terminal attached: keeping them (re-run with --remove-old to reclaim the space)"; return 0 ;; esac
  fi
  if [ "$do_rm" = 0 ]; then log "keeping them (./install.sh --remove-old reclaims the space later)"; return 0; fi
  local repo="${IMAGE%%:*}"
  local cmd="docker images --format '{{.Repository}}:{{.Tag}}' | grep '^$repo:' | grep -vx '$IMAGE' | grep -v ':latest\$' | xargs -r docker rmi"
  if [ "$do_imgs" = 1 ]; then
    [ -n "$head_imgs" ] && { bash -c "$cmd" 2>&1 | sed 's/^/  head: /' || true; }
    [ -n "$worker_imgs" ] && { WSSH "$cmd" 2>&1 | sed 's/^/  worker: /' || true; }
  fi
  if [ "$do_dfl" = 1 ] && [ -n "$dfl" ]; then
    rm_dir_any_owner bash -c; rm_dir_any_owner WSSH
  fi
  ok "leftovers removed"
}
# ---------- §3 weights -------------------------------------------------------
dl_in_container() { # dl_in_container <runner...> -- <hf_repo> <host_dir> [revision]
  local run=() a
  while [ "$1" != "--" ]; do run+=("$1"); shift; done; shift
  local repo=$1 dir=$2 rev=${3:-}
  local revarg=""
  [ -n "$rev" ] && revarg=", revision='$rev'"
  # Runs as the invoking user (not root) so the files can be removed later
  # without a container; the HF cache lives inside the target directory.
  "${run[@]}" "mkdir -p '$dir/.hf' && docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -e HF_HOME=/dl/.hf -v '$dir:/dl' --entrypoint python3 '$IMAGE' -c \"from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='/dl'$revarg)\""
}
have_exl3() { # have_exl3 <runner...> -- <dir>
  local run=() ; while [ "$1" != "--" ]; do run+=("$1"); shift; done; shift
  "${run[@]}" "test -f '$1/config.json' && [ \$(ls '$1'/*.safetensors 2>/dev/null | wc -l) -ge 120 ]"
}
have_dflash() { # have_dflash <runner...> -- <dir>   (config + a complete single-file checkpoint)
  local run=() ; while [ "$1" != "--" ]; do run+=("$1"); shift; done; shift
  "${run[@]}" "test -f '$1/config.json' && test -f '$1/model.safetensors' && [ \$(stat -c %s '$1/model.safetensors' 2>/dev/null || echo 0) -ge 1000000000 ]"
}
get_dflash() { # get_dflash <where> <runner...>
  local where=$1; shift
  have_dflash "$@" -- "$DFLASH_DIR" && return 0
  if "$@" "test -f '$DFLASH_DIR/config.json'"; then
    warn "$where: drafter download at $DFLASH_DIR is incomplete (model.safetensors missing or short) — resuming"
  fi
  dl_in_container "$@" -- "$DFLASH_REPO" "$DFLASH_DIR" "$DFLASH_REVISION" || die "$where: drafter download failed ($DFLASH_REPO)"
  have_dflash "$@" -- "$DFLASH_DIR" || die "$where: drafter still incomplete after download at $DFLASH_DIR"
}

download_models() {
  [ "$SKIP_DOWNLOAD" = 1 ] && { log "downloads skipped"; return; }
  echo
  log "DFlash2 drafter ($DFLASH_REPO) is CC BY-NC-ND 4.0 (research/eval)."
  log "It is downloaded from its source repository and never redistributed."
  echo
  # drafter: needed on BOTH boxes (each TP rank loads it)
  get_dflash worker WSSH
  get_dflash head bash -c

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
render_env() { # render_env <target: head|worker>  -> serve config on stdout
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
  printf '%s\n' "$out"
}
write_env() { # write_env <target: head|worker>
  if [ "$1" = head ]; then render_env head > "$HOME/.glm53-serve.env"
  else render_env worker | WSSH "cat > \$HOME/.glm53-serve.env"; fi
}

# Keep the previous release's files (both boxes) so a failed upgrade can be
# undone: serve config, launcher, warm-up, release stamp. Pre-stamp installs
# get a synthesized stamp carrying the image they were running.
PREV_FILES=(.glm53-serve.env launch-glm53-vllm-tp2.sh glm53-warmup.sh .glm53-kit-release)
save_prev() {
  [ "$UPGRADE" = 1 ] || return 0
  local f
  for f in "${PREV_FILES[@]}"; do
    [ -f "$HOME/$f" ] && cp -f "$HOME/$f" "$HOME/$f.prev"
    WSSH "[ -f \$HOME/$f ] && cp -f \$HOME/$f \$HOME/$f.prev; true"
  done
  if [ ! -f "$HOME/.glm53-kit-release.prev" ] && [ -n "$PREV_IMAGE" ]; then
    printf 'KIT_IMAGE=%s\nKIT_DATE=unknown\nKIT_COMMIT=pre-stamp\n' "$PREV_IMAGE" > "$HOME/.glm53-kit-release.prev"
    WSSH "cat > \$HOME/.glm53-kit-release.prev" < "$HOME/.glm53-kit-release.prev"
  fi
  ok "previous release saved as *.prev on both boxes (./install.sh --rollback restores it)"
}
write_stamp() {
  local dig commit content
  dig=$(docker image inspect --format '{{join .RepoDigests ","}}' "$IMAGE" 2>/dev/null)
  [ -n "$dig" ] || dig=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null)
  commit=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
  content=$(printf 'KIT_IMAGE=%s\nKIT_DIGEST=%s\nKIT_COMMIT=%s\nKIT_DATE=%s\nKIT_WEIGHTS_MODE=%s\nKIT_DFLASH_DIR=%s\nKIT_DFLASH_REPO=%s\nKIT_DFLASH_REVISION=%s' \
    "$IMAGE" "$dig" "$commit" "$(date -u +%Y-%m-%dT%H:%MZ)" "$WEIGHTS_MODE" "$DFLASH_DIR" "$DFLASH_REPO" "$DFLASH_REVISION")
  printf '%s\n' "$content" > "$STAMP"
  printf '%s\n' "$content" | WSSH "cat > \$HOME/.glm53-kit-release"
}

install_scripts() {
  save_prev
  write_env head; write_env worker
  sed "s/^KIT_VERSION=.*/KIT_VERSION=\"\${KIT_VERSION:-$KIT_VERSION}\"/" "$SCRIPT_DIR/scripts/launch-glm53-vllm-tp2.sh" > "$HOME/launch-glm53-vllm-tp2.sh"
  chmod 0755 "$HOME/launch-glm53-vllm-tp2.sh"
  install -m 0755 "$SCRIPT_DIR/scripts/glm53-warmup.sh" "$HOME/glm53-warmup.sh"
  scp -q -o BatchMode=yes "$HOME/launch-glm53-vllm-tp2.sh" "$WORKER:launch-glm53-vllm-tp2.sh"
  WSSH "chmod +x \$HOME/launch-glm53-vllm-tp2.sh"
  ok "launch scripts + serve config installed on both boxes"
}

# ---------- §4b upgrade summary (before anything is changed) -----------------
upgrade_summary() {
  local kit_commit img_state exl dfl envdiff old_imgs n
  kit_commit=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
  echo; log "===== what this run will do ====="
  if [ -n "$PREV_DESC" ]; then echo "  installed: $PREV_DESC"; else echo "  installed: nothing found (fresh install)"; fi
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then img_state="present on the head"; else img_state="pull ~18 GiB"; fi
  [ "$SKIP_PULL" = 1 ] && img_state="$img_state, pull skipped"
  echo "  target:    ${IMAGE##*:} (kit $kit_commit); image: $img_state; worker synced by digest"
  if [ "$SKIP_DOWNLOAD" = 1 ]; then
    echo "  weights:   downloads skipped (--skip-download)"
  else
    if [ "$WEIGHTS_MODE" = local ]; then
      exl="EXL3 head: $(have_exl3 bash -c -- "$MODEL_HOST_PATH" && echo present || echo 'download ~176 GiB'); worker: $(have_exl3 WSSH -- "$MODEL_HOST_PATH" && echo present || echo 'copy from head over the rail')"
    else
      exl="EXL3 worker: $(have_exl3 WSSH -- "$MODEL_HOST_PATH" && echo present || echo 'download ~176 GiB') (NFS to the head)"
    fi
    dfl="drafter $DFLASH_DIR head: $(have_dflash bash -c -- "$DFLASH_DIR" && echo present || echo 'download 1.3 GiB'); worker: $(have_dflash WSSH -- "$DFLASH_DIR" && echo present || echo 'download 1.3 GiB')"
    echo "  weights:   $exl"
    echo "             $dfl"
  fi
  if [ -f "$HOME/.glm53-serve.env" ]; then
    envdiff=$(diff <(grep -v '^# Written' "$HOME/.glm53-serve.env") <(render_env head | grep -v '^# Written') | grep -E '^[<>]' | sed 's/^< /    - /; s/^> /    + /')
    if [ -n "$envdiff" ]; then echo "  serve config (~/.glm53-serve.env) changes:"; echo "$envdiff"; else echo "  serve config: unchanged"; fi
    echo "  scripts:   launcher + warm-up replaced on both boxes (previous copies kept as *.prev)"
  else
    echo "  serve config + scripts: written fresh on both boxes"
  fi
  old_imgs=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^${IMAGE%%:*}:" | grep -vx "$IMAGE" | grep -v ':latest$' | sed 's/.*://' | tr '\n' ' ')
  if [ -n "$old_imgs" ]; then
    echo "  leftovers: superseded kit images on the head: $old_imgs(~18 GiB each; removal is offered after a successful start, or --remove-old / --keep-old)"
  fi
  if [ -d "$HOME/models/glm53-dflash2" ] && [ "$DFLASH_DIR" != "$HOME/models/glm53-dflash2" ]; then
    echo "  leftovers: ~/models/glm53-dflash2 (bf16 drafter, 2.3 GiB) is no longer used by this release (removal offered after a successful start)"
  fi
  if [ ${#ENV_NOTES[@]} -gt 0 ]; then
    echo "  .env notes:"; for n in "${ENV_NOTES[@]}"; do echo "    - $n"; done
  fi
  [ "$UPGRADE" = 1 ] && echo "  safety:    if the new release does not come up or fails its smoke test, the previous release is restored (./install.sh --rollback does the same later)"
  echo
  [ "$YES" = 1 ] && return 0
  local ans=""
  if ( : < /dev/tty ) 2>/dev/null; then
    printf 'Continue? [Y/n] ' > /dev/tty; read -r ans < /dev/tty || true
    case "$ans" in n|N|no|NO) die "aborted — nothing was changed" ;; esac
  else
    log "no terminal attached: proceeding (--yes silences this note)"
  fi
}

# ---------- §5 launch --------------------------------------------------------
# boot_and_wait: tear down, launch worker then head, wait for the API.
# Returns 1 (with the diagnostics printed) instead of exiting, so an upgrade
# can fall back to the previous release.
boot_and_wait() {
  # Tear down a live head FIRST: a fresh worker otherwise rendezvouses with
  # the OLD head's TCP store and dies of connection-reset the moment that
  # head is replaced (leaving the new head waiting on a dead worker).
  docker rm -f vllm_glm53 >/dev/null 2>&1 || true
  log "launching worker (rank 1)"
  WSSH "\$HOME/launch-glm53-vllm-tp2.sh 1" || { warn "worker launch failed"; return 1; }
  sleep 30
  log "launching head (rank 0); weight load + engine init takes ~4 min (direct-I/O loader) to ~13 min"
  "$HOME/launch-glm53-vllm-tp2.sh" 0 || { warn "head launch failed"; return 1; }
  local t0 elapsed worker_strikes=0
  t0=$(date +%s)
  until curl -sf -m5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
    sleep 15
    docker ps --format '{{.Names}}' | grep -q '^vllm_glm53$' || {
      echo; docker logs vllm_glm53 2>&1 | grep -E "Error|Traceback|assert|error:" | tail -12 | sed 's/^/    head: /'
      warn "head container exited during boot (engine errors above; full log: docker logs vllm_glm53). If it wedged or OOMed at ~90% of shard load: grow swap to >=32 GiB on both boxes, or set LOAD_FORMAT=instanttensor in .env."
      return 1
    }
    # A dead worker otherwise burns the full 30-min timeout while the head
    # waits on rendezvous. 3 consecutive strikes so one flaky ssh probe
    # can't kill a healthy boot.
    if WSSH "docker ps --format '{{.Names}}' | grep -q '^vllm_glm53\$'" >/dev/null 2>&1; then
      worker_strikes=0
    else
      worker_strikes=$(( worker_strikes + 1 ))
      if [ "$worker_strikes" -ge 3 ]; then
        echo; WSSH "docker logs vllm_glm53 2>&1 | grep -E 'Error|Traceback|assert|error:' | tail -12" | sed 's/^/    worker: /' || true
        warn "worker container is gone during boot (worker engine errors above, if any)"
        return 1
      fi
    fi
    elapsed=$(( $(date +%s) - t0 ))
    [ "$elapsed" -gt 1800 ] && { warn "API not up after 30 min — docker logs vllm_glm53"; return 1; }
  done
  ok "API up after $(( $(date +%s) - t0 ))s"
  log "warming JIT shapes (~20 s)"
  API_BASE="http://localhost:$PORT" bash "$HOME/glm53-warmup.sh" || warn "warmup had failures (serving continues)"
  return 0
}
smoke_test() {
  local smoke
  smoke=$(curl -s -m 90 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}],"temperature":0,"max_tokens":200}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
  case "$smoke" in
    *391*) ok "smoke test passed: '$smoke'"; return 0 ;;
  esac
  warn "smoke test failed (got: '$smoke'); image: $(docker inspect vllm_glm53 --format '{{.Config.Image}}' 2>/dev/null)"
  docker logs vllm_glm53 2>&1 | grep -E "Error|Traceback|assert|error:" | tail -8 | sed 's/^/    head: /' >&2
  return 1
}
rollback() { # restore the *.prev files on both boxes and relaunch the previous release
  local f prev_img
  [ -f "$HOME/.glm53-serve.env.prev" ] || { warn "nothing to roll back to (no *.prev files from an earlier upgrade)"; return 1; }
  prev_img=$(sed -n 's/^KIT_IMAGE=//p' "$HOME/.glm53-kit-release.prev" 2>/dev/null | head -1)
  log "restoring the previous release${prev_img:+ (${prev_img##*:})} on both boxes"
  for f in "${PREV_FILES[@]}"; do
    [ -f "$HOME/$f.prev" ] && cp -f "$HOME/$f.prev" "$HOME/$f"
    WSSH "[ -f \$HOME/$f.prev ] && cp -f \$HOME/$f.prev \$HOME/$f; true"
  done
  if [ -n "$prev_img" ] && ! docker image inspect "$prev_img" >/dev/null 2>&1; then
    warn "previous image $prev_img is not on the head any more — pulling it"
    docker pull "$prev_img" || return 1
  fi
  boot_and_wait || return 1
  smoke_test || return 1
  ok "previous release restored and serving${prev_img:+: ${prev_img##*:}}"
}
start_server() {
  [ "$NO_START" = 1 ] && { log "--no-start: skipping launch. Worker: ./launch-glm53-vllm-tp2.sh 1, then head: ./launch-glm53-vllm-tp2.sh 0"; return; }
  if boot_and_wait && smoke_test; then
    write_stamp
    ok "release stamp written (~/.glm53-kit-release on both boxes)"
    return 0
  fi
  if [ "$UPGRADE" = 1 ] && [ "$NO_ROLLBACK" = 0 ] && [ -f "$HOME/.glm53-serve.env.prev" ]; then
    warn "the new release (${IMAGE##*:}) did not come up cleanly — restoring the previous one"
    if rollback; then
      die "upgrade to ${IMAGE##*:} failed; the previous release is back and serving. Fix the cause shown above and re-run (--no-rollback keeps a failed boot running for debugging)."
    fi
    die "upgrade failed AND the previous release did not come back — see the logs above; tear down: docker rm -f vllm_glm53; ssh $WORKER docker rm -f vllm_glm53"
  fi
  warn "containers are left running so the logs survive; tear down: docker rm -f vllm_glm53; ssh $WORKER docker rm -f vllm_glm53"
  die "start failed"
}

if [ "$ROLLBACK" = 1 ]; then
  rollback || die "rollback failed"
  exit 0
fi
if [ "$CLEANUP" = 1 ]; then
  WSSH true || die "passwordless SSH to $WORKER failed"
  offer_cleanup
  exit 0
fi
lint_env
verify_hosts
detect_installed
upgrade_summary
pull_image
download_models
install_scripts
start_server
offer_cleanup

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
