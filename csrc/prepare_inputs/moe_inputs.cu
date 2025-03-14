/*
 This file has the implementations for the helper method
 for cutlass based grouped gemm kernels to get the required
 problem sizes, expert offsetsl and maps.

*/

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
namespace vllm_cutlass_moe{

// basic correctness, currently unused, run with <<<1, num_experts>>>
__global__ void get_grouped_mm_data_kernel(
    const int* __restrict__ topk_ids, int32_t* expert_offsets,
    int32_t* problem_sizes1, int32_t* problem_sizes2, int32_t* arg_sort,
    int32_t* arg_sort_prim, int topk_length, int n, int k, int topk) {
  int expert_id = threadIdx.x;
  int num_experts = blockDim.x;

  int occurrences = 0;
  for (int i = 0; i < topk_length; ++i) {
    occurrences += (topk_ids[i] == expert_id);
  }
  problem_sizes1[expert_id * 3] = occurrences;
  problem_sizes1[expert_id * 3 + 1] = 2 * n;
  problem_sizes1[expert_id * 3 + 2] = k;
  problem_sizes2[expert_id * 3] = occurrences;
  problem_sizes2[expert_id * 3 + 1] = k;
  problem_sizes2[expert_id * 3 + 2] = n;
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t tot_offset = 0;
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      tot_offset += problem_sizes1[i * 3];
      expert_offsets[i + 1] = tot_offset;
    }
  }

  __syncthreads();

  int start = expert_offsets[expert_id];
  int end = expert_offsets[expert_id + 1];
  for (int i = 0; i < topk_length; ++i) {
    if (topk_ids[i] == expert_id) {
      arg_sort[start] = i / topk;
      arg_sort_prim[i] = start;
      ++start;
      if (start == end) {
        break;
      }
    }
  }
}

constexpr int THREADS_PER_EXPERT = 512;

__global__ void compute_problem_sizes(const int* __restrict__ topk_ids,
                                      int32_t* problem_sizes1,
                                      int32_t* problem_sizes2,
                                      int32_t* atomic_buffer,
                                      const int topk_length, const int n,
                                      const int k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = 2 * n;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = k;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, const int num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }

}

__global__ void compute_arg_sorts(const int* __restrict__ topk_ids,
                                  int32_t* arg_sort, int32_t* arg_sort_prim,
                                  int32_t* atomic_buffer, const int topk_length,
                                  const int topk) {
  int expert_id = blockIdx.x;

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    if (topk_ids[i] == expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      arg_sort[start] = i / topk;
      arg_sort_prim[i] = start;
    }
  }
}

constexpr int THREADS_PER_EXPERT_2 = 32;

// 1 warp per expert
// 4 experts per block
__global__ void compute_problem_sizes_multi_expert(
    const int* __restrict__ topk_ids, int32_t* problem_sizes1,
    int32_t* problem_sizes2, int32_t* atomic_buffer, const int topk_length,
    const int n, const int k) {
  int expert_id = blockIdx.x * 4 + threadIdx.x / THREADS_PER_EXPERT_2;
  int start = threadIdx.x % THREADS_PER_EXPERT_2;

  int occurrences = 0;
  for (int i = start; i < topk_length; i += THREADS_PER_EXPERT_2) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  // we only need this if #threads/expert > warp_size
  if constexpr (THREADS_PER_EXPERT_2 > 32) {
    __syncthreads();
  }

  if (start == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = 2 * n;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = k;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

__global__ void compute_arg_sorts_multi_expert(
    const int* __restrict__ topk_ids, int32_t* arg_sort, int32_t* arg_sort_prim,
    int32_t* atomic_buffer, const int topk_length, const int topk) {
  int expert_id = blockIdx.x * 4 + threadIdx.x / THREADS_PER_EXPERT_2;
  int start = threadIdx.x % THREADS_PER_EXPERT_2;

  for (int i = start; i < topk_length; i += THREADS_PER_EXPERT_2) {
    if (topk_ids[i] == expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      arg_sort[start] = i / topk;
      arg_sort_prim[i] = start;
    }
  }
}

void get_grouped_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& arg_sort, torch::Tensor& arg_sort_prim,
    const int64_t num_experts, const int64_t n, const int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  // TODO this is an alternative way to block kernels

  constexpr bool multi_expert_blocks = false;
  if constexpr (multi_expert_blocks) {
    int num_blocks = (num_experts + 3) / 4;
    int num_threads = THREADS_PER_EXPERT_2 * 4;
    compute_problem_sizes_multi_expert<<<num_blocks, num_threads, 0, stream>>>(
        (const int32_t*)topk_ids.data_ptr(),
        (int32_t*)problem_sizes1.data_ptr(),
        (int32_t*)problem_sizes2.data_ptr(), (int32_t*)atomic_buffer.data_ptr(),
        topk_ids.numel(), n, k);
    compute_expert_offsets<<<1, 1, 0, stream>>>(
        (const int32_t*)problem_sizes1.data_ptr(),
        (int32_t*)expert_offsets.data_ptr(), (int32_t*)atomic_buffer.data_ptr(),
        num_experts);
    compute_arg_sorts_multi_expert<<<num_blocks, num_threads, 0, stream>>>(
        (const int32_t*)topk_ids.data_ptr(), (int32_t*)arg_sort.data_ptr(),
        (int32_t*)arg_sort_prim.data_ptr(), (int32_t*)atomic_buffer.data_ptr(),
        topk_ids.numel(), topk_ids.size(1));
    return;
  }

  int num_threads = THREADS_PER_EXPERT < topk_ids.numel() ? THREADS_PER_EXPERT : topk_ids.numel() ;
  compute_problem_sizes<<<num_experts, num_threads, 0, stream>>>(
      (const int32_t*)topk_ids.data_ptr(), (int32_t*)problem_sizes1.data_ptr(),
      (int32_t*)problem_sizes2.data_ptr(), (int32_t*)atomic_buffer.data_ptr(),
      topk_ids.numel(), n, k);
  compute_expert_offsets<<<1, 1, 0, stream>>>(
      (const int32_t*)problem_sizes1.data_ptr(),
      (int32_t*)expert_offsets.data_ptr(), (int32_t*)atomic_buffer.data_ptr(),
      num_experts);
  compute_arg_sorts<<<num_experts, num_threads, 0, stream>>>(
      (const int32_t*)topk_ids.data_ptr(), (int32_t*)arg_sort.data_ptr(),
      (int32_t*)arg_sort_prim.data_ptr(), (int32_t*)atomic_buffer.data_ptr(),
      topk_ids.numel(), topk_ids.size(1));
}


} // namespace vllm_cutlass_moe

void get_grouped_mm_data(const torch::Tensor& topk_ids,
                         torch::Tensor& expert_offsets,
                         torch::Tensor& problem_sizes1,
                         torch::Tensor& problem_sizes2, torch::Tensor& arg_sort,
                         torch::Tensor& arg_sort_prim,
                         const int64_t num_experts, const int64_t n,
                         const int64_t k) {
  vllm_cutlass_moe::get_grouped_mm_data_caller(topk_ids, expert_offsets, problem_sizes1,
                             problem_sizes2, arg_sort, arg_sort_prim,
                             num_experts, n, k);
}