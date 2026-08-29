# topk_fix — persistent_topk full-smem retry for the v1 image

The shipped `v1-dflash2` image's `_C.persistent_topk` fails at long
declarations on GB10 (48 SMs, ~99 KB smem/block): with ≤8 decode rows its
small-batch smem tier oversubscribes the cooperative grid (62 CTAs > 48 at a
524k stride drafterless; 90 > 48 at 1M for ANY solo-decoding request). The
fix (fork commit `c14e6f443`) retries the launch computation at the full
per-block smem before failing; the next image bakes it into `_C`. This
directory builds the same fixed launcher as a standalone extension for the
current image.

Build once per box (the serving container has nvcc):

```bash
docker cp tools/topk_fix vllm_glm53:/tmp/topk_fix
docker exec vllm_glm53 python3 /tmp/topk_fix/build_topk_fix.py /cache/topk_fix
cp ~/glm53-vllm-cache/topk_fix/topk_fix.so ~/glm53-vllm-cache/topk_fix.so
```

Then launch with `GLM53_TOPK_FIX_SO=/cache/topk_fix.so` (the launcher
forwards it; the engine logs `persistent_topk routed through ...` at boot).
Required for `MAX_LEN` above ~358k drafterless or above 524k with the
drafter; harmless otherwise. The kernel (.cuh) is byte-identical to the
image's — only the host-side launch heuristic differs.
