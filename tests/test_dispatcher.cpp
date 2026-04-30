// filename: test_dispatcher.cpp
// purpose: Validates runtime dispatcher backend selection and output parity.
// phase: Phase 3 - Runtime Abstraction
// last modified: 2026-04-29

#include "backends/cpu/cpu_backend.h"
#include "backends/cuda/cuda_backend.h"
#include "runtime/dispatcher.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {

constexpr std::size_t kDim = 4U;
constexpr float kTolerance = 1.0e-5F;

/**
 * @brief Computes max absolute difference between same-sized vectors.
 * @param lhs First vector.
 * @param rhs Second vector.
 * @return Maximum absolute element-wise difference.
 * @sideeffects None.
 */
float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = std::fabs(lhs[i] - rhs[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

}  // namespace

int main() {
  const std::vector<float> a{
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
      9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
  const std::vector<float> b{
      16.0F, 15.0F, 14.0F, 13.0F, 12.0F, 11.0F, 10.0F, 9.0F,
      8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};

  auto cpu_backend = std::make_shared<opengpu::backends::cpu::CPUBackend>();
  auto cuda_backend = std::make_shared<opengpu::backends::cuda::CUDABackend>();

  cpu_backend->set_pending_matmul_inputs(a, b, kDim);
  cuda_backend->set_pending_matmul_inputs(a, b, kDim);

  opengpu::runtime::Dispatcher dispatcher;
  if (!dispatcher.register_backend("cpu", cpu_backend)) {
    std::cerr << "Failed to register cpu backend\n";
    return EXIT_FAILURE;
  }
  if (!dispatcher.register_backend("cuda", cuda_backend)) {
    std::cerr << "Failed to register cuda backend\n";
    return EXIT_FAILURE;
  }

  const opengpu::runtime::Kernel kernel{"matmul", 1U};
  const opengpu::runtime::LaunchConfig launch{
      opengpu::runtime::Dim3{1U, 1U, 1U},
      opengpu::runtime::Dim3{16U, 16U, 1U},
  };

  if (!dispatcher.dispatch("cpu", kernel, launch)) {
    std::cerr << "Dispatch failed for backend: cpu\n";
    return EXIT_FAILURE;
  }
  std::cout << "Backend used: cpu\n";

  if (!dispatcher.dispatch("cuda", kernel, launch)) {
    std::cerr << "Dispatch failed for backend: cuda\n";
    return EXIT_FAILURE;
  }
  std::cout << "Backend used: cuda\n";

  const std::vector<float> cpu_output = cpu_backend->last_output();
  const std::vector<float> cuda_output = cuda_backend->last_output();
  const float diff = max_abs_diff(cpu_output, cuda_output);

  std::cout << std::fixed << std::setprecision(6)
            << "Dispatcher cpu vs cuda max abs diff: " << diff << '\n';
  if (diff > kTolerance) {
    std::cerr << "Dispatcher mismatch exceeds tolerance\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
