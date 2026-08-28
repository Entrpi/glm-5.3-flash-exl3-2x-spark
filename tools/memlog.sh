#!/usr/bin/env bash
# 1 Hz MemAvailable sampler — the memory-floor methodology behind the KV
# budget in this recipe. Run on EACH box while driving load (e.g.
# scripts/saturation_bench.py + a 112k-token prompt), then read the minimum:
#   ./memlog.sh /tmp/memlog &          # start before the load
#   sort -n /tmp/memlog | head -1      # floor in KiB after the run
# Keep the floor above ~4 GiB on the memory-binding box before raising
# KV_CACHE_MEMORY or GMU — explicit KV budgets bypass vLLM's profiling
# reserve, and the failure mode on GB10 unified memory is a swap wedge,
# not a graceful OOM. Floors also age ~1.5-2 GiB per day of workload;
# fresh-boot floors are optimistic.
OUT="${1:-/tmp/memlog}"
: > "$OUT"
while true; do
  awk '/MemAvailable/ {print $2}' /proc/meminfo >> "$OUT"
  sleep 1
done
