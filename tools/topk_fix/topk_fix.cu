// Standalone build of the fixed persistent_topk launcher (fork commit
// c14e6f443: retry with full smem before declaring oversubscription).
//
// Interim deployment vehicle for the v1-dflash2 image, whose baked _C
// predates the fix: the kernel (.cuh) is byte-identical to the image's;
// only the host-side launch heuristic differs. Registered as
// torch.ops.topk_fix.persistent_topk with the same schema as
// _C::persistent_topk; the overlay-patched sparse_attn_indexer_kpool.py
// routes to it when GLM53_TOPK_FIX_SO is set. Retired when the v2 image
// bakes the fixed _C.
//
// Build in-container with build_topk_fix.py (torch cpp_extension, arch
// auto-detected from the GPU).

#include <cuda_runtime.h>

#include <algorithm>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "persistent_topk.cuh"

namespace {

const cudaDeviceProp* get_device_prop_cached() {
  static cudaDeviceProp prop;
  static bool initialized = false;
  if (!initialized) {
    int dev = -1;
    TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "cudaGetDevice failed");
    TORCH_CHECK(cudaGetDeviceProperties(&prop, dev) == cudaSuccess,
                "cudaGetDeviceProperties failed");
    initialized = true;
  }
  return &prop;
}

template <int TopK>
void launch_persistent_topk(const at::Tensor& logits, const at::Tensor& lengths,
                            at::Tensor& output, const at::Tensor& workspace,
                            int64_t max_seq_len) {
  namespace P = vllm::persistent;

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.stride(0);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  static int num_sms = 0;
  static int max_smem_per_block = 0;
  if (num_sms == 0) {
    const cudaDeviceProp* device_prop = get_device_prop_cached();
    num_sms = device_prop->multiProcessorCount;
    max_smem_per_block = device_prop->sharedMemPerBlockOptin;
  }

  TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
  TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must be uint8");

  // Small batches cap smem to keep more than one resident CTA per SM. At
  // very long strides the capped chunk can oversubscribe the cooperative
  // grid, and on parts with <128KB smem per block (e.g. GB10/sm121 at
  // ~99KB) there is no fallback — so before giving up, retry the launch
  // computation once with the full per-block smem, which shrinks
  // ctas_per_group roughly in proportion.
  int effective_max_smem;
  if (num_rows <= 4) {
    effective_max_smem =
        std::min(max_smem_per_block, static_cast<int>(P::kSmemMedium));
  } else if (num_rows <= 8) {
    constexpr int kSmemCapMedium = 48 * 1024;
    effective_max_smem = std::min(max_smem_per_block, kSmemCapMedium);
  } else {
    effective_max_smem = max_smem_per_block;
  }

  uint32_t vec_size = 1;
  if (stride % 4 == 0)
    vec_size = 4;
  else if (stride % 2 == 0)
    vec_size = 2;

  const bool needs_cooperative =
      static_cast<uint32_t>(max_seq_len) > P::RADIX_THRESHOLD;

  uint32_t ctas_per_group = 0;
  uint32_t chunk_size = 0;
  uint32_t num_groups = 0;
  uint32_t total_ctas = 0;
  uint32_t hw_resident_cap = 0;
  size_t smem_size = 0;
  for (;;) {
    size_t available_for_ordered =
        static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
    uint32_t max_chunk_elements =
        static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

    max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
    uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
    if (max_chunk_elements < min_chunk) max_chunk_elements = min_chunk;

    ctas_per_group = (static_cast<uint32_t>(stride) + max_chunk_elements - 1) /
                     max_chunk_elements;
    chunk_size =
        (static_cast<uint32_t>(stride) + ctas_per_group - 1) / ctas_per_group;
    chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
    if (chunk_size > max_chunk_elements) chunk_size = max_chunk_elements;

    smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
    if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;

    // Query occupancy for the instantiation that will actually launch;
    // overestimating it deadlocks the cooperative barrier.
    int occupancy = 1;
    cudaError_t occ_err = cudaSuccess;
    if (vec_size == 4) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 4>, P::kThreadsPerBlock,
          smem_size);
    } else if (vec_size == 2) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 2>, P::kThreadsPerBlock,
          smem_size);
    } else {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 1>, P::kThreadsPerBlock,
          smem_size);
    }
    TORCH_CHECK(occ_err == cudaSuccess,
                "persistent_topk occupancy query failed: ",
                cudaGetErrorString(occ_err));
    if (occupancy < 1) occupancy = 1;

    hw_resident_cap =
        static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
    uint32_t max_resident_ctas = hw_resident_cap;
    if (needs_cooperative) {
      // Reserve one CTA per SM when occupancy allows; fall back to a single
      // CTA when occupancy == 1 (the most deadlock-prone case — any
      // straggler kernel that takes the only slot on one SM hangs the
      // barrier). Never drop below one full group's worth.
      uint32_t headroom = (occupancy > 1) ? static_cast<uint32_t>(num_sms) : 1u;
      if (max_resident_ctas >= headroom + ctas_per_group) {
        max_resident_ctas -= headroom;
      }
    }
    num_groups = std::min(max_resident_ctas / ctas_per_group,
                          static_cast<uint32_t>(num_rows));
    if (num_groups == 0) num_groups = 1;
    total_ctas = num_groups * ctas_per_group;

    if (!(needs_cooperative && total_ctas >= hw_resident_cap)) break;  // >=: zero-spare-slot launches count as oversubscribed (straggler deadlock)
    if (effective_max_smem < max_smem_per_block) {
      effective_max_smem = max_smem_per_block;
      continue;
    }
    break;  // Oversubscribed even at full smem; fail below.
  }

  // No FilteredTopK in this mini-build (it needs >=128KB smem, which the
  // sm121 target this extension exists for does not have). Fail loudly.
  TORCH_CHECK(!(needs_cooperative && total_ctas >= hw_resident_cap),
              "topk_fix.persistent_topk would oversubscribe even at full "
              "smem: total_ctas=",
              total_ctas, " > num_sms*occupancy=", hw_resident_cap,
              " (TopK=", TopK, ", vec_size=", vec_size,
              ", ctas_per_group=", ctas_per_group, ", smem=", smem_size, ")");

  size_t state_bytes = num_groups * sizeof(P::RadixRowState);
  TORCH_CHECK(workspace.size(0) >= static_cast<int64_t>(state_bytes),
              "workspace too small, need ", state_bytes, " bytes");

  // Zero the per-group RadixRowState region before launch. Issued
  // unconditionally so cudagraph capture always records the memset node
  // (same rationale as the in-tree launcher).
  {
    cudaError_t mz_err = cudaMemsetAsync(workspace.data_ptr<uint8_t>(), 0,
                                         state_bytes, stream);
    TORCH_CHECK(mz_err == cudaSuccess,
                "row_states memset failed: ", cudaGetErrorString(mz_err));
  }

  P::PersistentTopKParams params;
  params.input = logits.const_data_ptr<float>();
  params.output = output.mutable_data_ptr<int32_t>();
  params.lengths = lengths.const_data_ptr<int32_t>();
  params.num_rows = static_cast<uint32_t>(num_rows);
  params.stride = static_cast<uint32_t>(stride);
  params.top_k = static_cast<uint32_t>(TopK);
  params.chunk_size = chunk_size;
  params.row_states =
      reinterpret_cast<P::RadixRowState*>(workspace.data_ptr<uint8_t>());
  params.ctas_per_group = ctas_per_group;
  params.max_seq_len = static_cast<uint32_t>(max_seq_len);

#define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                     \
  do {                                                                      \
    auto kernel = &P::persistent_topk_kernel<TOPK_VAL, VS>;                 \
    cudaError_t err = cudaFuncSetAttribute(                                 \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);    \
    TORCH_CHECK(err == cudaSuccess,                                         \
                "Failed to set smem: ", cudaGetErrorString(err));           \
    kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params); \
  } while (0)

  if (vec_size == 4) {
    LAUNCH_PERSISTENT(TopK, 4);
  } else if (vec_size == 2) {
    LAUNCH_PERSISTENT(TopK, 2);
  } else {
    LAUNCH_PERSISTENT(TopK, 1);
  }
#undef LAUNCH_PERSISTENT

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "persistent_topk failed: ", cudaGetErrorString(err));
}

void persistent_topk_fix(const at::Tensor& logits, const at::Tensor& lengths,
                         at::Tensor output, const at::Tensor& workspace,
                         int64_t k, int64_t max_seq_len) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(logits.scalar_type() == at::kFloat, "Only float32 supported");
  TORCH_CHECK(lengths.scalar_type() == at::kInt, "lengths must be int32");
  TORCH_CHECK(output.scalar_type() == at::kInt, "output must be int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
              "lengths must be 1D or 2D");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");

  const int64_t num_rows = logits.size(0);

  TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
              "output size mismatch");
  TORCH_CHECK(k == 512 || k == 1024 || k == 2048,
              "persistent_topk supports k=512, k=1024, or k=2048, got k=", k);

  if (k == 512) {
    launch_persistent_topk<512>(logits, lengths, output, workspace,
                                max_seq_len);
  } else if (k == 1024) {
    launch_persistent_topk<1024>(logits, lengths, output, workspace,
                                 max_seq_len);
  } else {
    launch_persistent_topk<2048>(logits, lengths, output, workspace,
                                 max_seq_len);
  }
}

}  // anonymous namespace

TORCH_LIBRARY(topk_fix, m) {
  m.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor(a!) output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
}

TORCH_LIBRARY_IMPL(topk_fix, CUDA, m) {
  m.impl("persistent_topk", &persistent_topk_fix);
}
