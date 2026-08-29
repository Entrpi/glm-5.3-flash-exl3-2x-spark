#!/usr/bin/env python3
"""Build topk_fix.so inside the serving container.

Usage (from the box, container running):
  docker cp topk_fix/ vllm_glm53:/tmp/topk_fix/
  docker exec vllm_glm53 python3 /tmp/topk_fix/build_topk_fix.py /cache/topk_fix
  # -> /cache/topk_fix/topk_fix.so  (on the host: ~/glm53-vllm-cache/topk_fix/)

Then serve with GLM53_TOPK_FIX_SO=/cache/topk_fix/topk_fix.so (the patched
sparse_attn_indexer_kpool.py overlay loads it and routes persistent_topk
through torch.ops.topk_fix; unset -> stock _C op, unchanged behavior).
"""

import shutil
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

here = Path(__file__).resolve().parent
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else here
out_dir.mkdir(parents=True, exist_ok=True)

build_dir = out_dir / "build"
build_dir.mkdir(parents=True, exist_ok=True)
load(
    name="topk_fix",
    sources=[str(here / "topk_fix.cu")],
    extra_cuda_cflags=["-O3"],
    build_directory=str(build_dir),
    verbose=True,
    is_python_module=False,
)

so = next(build_dir.glob("topk_fix.so"))
dest = out_dir / "topk_fix.so"
shutil.copy2(so, dest)
print(f"built: {dest}")

# The load() above already registered the library in this process; the
# copy at `dest` is what serving loads via GLM53_TOPK_FIX_SO.
dev = "cuda"
n, k, seqlen = 2, 512, 4096
logits = torch.randn(n, seqlen, device=dev, dtype=torch.float32)
lengths = torch.full((n,), seqlen, device=dev, dtype=torch.int32)
out = torch.full((n, k), -1, device=dev, dtype=torch.int32)
workspace = torch.zeros(4 * 1024 * 1024, device=dev, dtype=torch.uint8)
torch.ops.topk_fix.persistent_topk(logits, lengths, out, workspace, k, seqlen)
torch.cuda.synchronize()
ref = torch.topk(logits, k, dim=-1).indices
got = out.long().sort(dim=-1).values
exp = ref.sort(dim=-1).values
assert torch.equal(got, exp), "topk_fix smoke mismatch vs torch.topk"
print("smoke: topk_fix.persistent_topk matches torch.topk on 2x4096/k512")
