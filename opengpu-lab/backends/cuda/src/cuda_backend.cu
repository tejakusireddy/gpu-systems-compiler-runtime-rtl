// filename: cuda_backend.cu
// purpose: Implements CUDA backend launch and matmul execution path.
// phase: Phase 2 - Real GPU
// last modified: 2026-04-29

#include "backends/cuda/cuda_backend.h"

#include <cstddef>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include "matmul.cuh"

namespace opengpu::backends::cuda {

bool CUDABackend::launch(const opengpu::runtime::Kernel& kernel,
                         const opengpu::runtime::LaunchConfig& launch) {
  constexpr const char* kMatmulKernelName = "matmul";
  const bool valid_launch = (launch.grid.x > 0U) && (launch.grid.y > 0U) &&
                            (launch.grid.z > 0U) && (launch.block.x > 0U) &&
                            (launch.block.y > 0U) && (launch.block.z > 0U);
  if (!valid_launch) {
    return false;
  }
  if (kernel.name != kMatmulKernelName) {
    return false;
  }
  if (pending_n_ == 0U) {
    return false;
  }
  if (pending_a_.size() != pending_n_ * pending_n_ ||
      pending_b_.size() != pending_n_ * pending_n_) {
    return false;
  }

  std::vector<float> output;
  const bool ok = run_matmul(pending_a_, pending_b_, pending_n_, &output);
  if (!ok) {
    return false;
  }
  last_output_ = std::move(output);
  return true;
}

bool CUDABackend::run_matmul(const std::vector<float>& a, const std::vector<float>& b,
                             const std::size_t n, std::vector<float>* out) const {
  if (out == nullptr || n == 0U) {
    return false;
  }
  const std::size_t expected_elements = n * n;
  if (a.size() != expected_elements || b.size() != expected_elements) {
    return false;
  }

#if defined(__CUDACC__)
  const std::size_t bytes = expected_elements * sizeof(float);
  float* a_device = nullptr;
  float* b_device = nullptr;
  float* c_device = nullptr;

  if (cudaMalloc(reinterpret_cast<void**>(&a_device), bytes) != cudaSuccess) {
    return false;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&b_device), bytes) != cudaSuccess) {
    cudaFree(a_device);
    return false;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&c_device), bytes) != cudaSuccess) {
    cudaFree(a_device);
    cudaFree(b_device);
    return false;
  }

  const bool copied_to_device =
      (cudaMemcpy(a_device, a.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess) &&
      (cudaMemcpy(b_device, b.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
  if (!copied_to_device) {
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    return false;
  }

  const bool launch_ok = kernels::launch_naive_matmul(a_device, b_device, c_device, n);
  if (!launch_ok) {
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    return false;
  }

  out->assign(expected_elements, 0.0F);
  const bool copied_to_host =
      (cudaMemcpy(out->data(), c_device, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);

  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);

  return copied_to_host;
#else
  out->assign(expected_elements, 0.0F);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      float accum = 0.0F;
      for (std::size_t k = 0; k < n; ++k) {
        accum += a[(row * n) + k] * b[(k * n) + col];
      }
      (*out)[(row * n) + col] = accum;
    }
  }
  // TODO(phase-2): Replace fallback with real CUDA execution on CUDA-enabled builds.
  return true;
#endif
}

void CUDABackend::set_pending_matmul_inputs(const std::vector<float>& a,
                                            const std::vector<float>& b,
                                            const std::size_t n) {
  pending_a_ = a;
  pending_b_ = b;
  pending_n_ = n;
}

std::vector<float> CUDABackend::last_output() const { return last_output_; }

}  // namespace opengpu::backends::cuda
