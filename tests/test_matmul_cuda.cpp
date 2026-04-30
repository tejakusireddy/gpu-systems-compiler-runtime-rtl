// filename: test_matmul_cuda.cpp
// purpose: Validates CUDA backend matmul output against CPU reference output.
// phase: Phase 2 - Real GPU
// last modified: 2026-04-29

#include "backends/cuda/cuda_backend.h"
#include "matmul.h"
#include "runtime/kernel.h"
#include "runtime/launch_config.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kSmallDim = 4U;
constexpr std::size_t kLargeDim = 64U;
constexpr float kTolerance = 1.0e-5F;

/**
 * @brief Builds deterministic matrix values for repeatable test coverage.
 * @param n Matrix dimension.
 * @param seed Seed offset for pattern generation.
 * @return Matrix data in row-major order.
 * @sideeffects None.
 */
std::vector<float> make_matrix(const std::size_t n, const std::size_t seed) {
  std::vector<float> out(n * n, 0.0F);
  constexpr std::size_t kValueMod = 23U;
  constexpr float kScale = 0.125F;

  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::size_t raw = (i + seed) % kValueMod;
    out[i] = static_cast<float>(raw) * kScale;
  }
  return out;
}

/**
 * @brief Returns the fixed 4x4 matrix values used by the CPU test.
 * @param None.
 * @return Matrix A in row-major order.
 * @sideeffects None.
 */
std::vector<float> make_small_a() {
  return {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
          9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
}

/**
 * @brief Returns the fixed 4x4 matrix values used by the CPU test.
 * @param None.
 * @return Matrix B in row-major order.
 * @sideeffects None.
 */
std::vector<float> make_small_b() {
  return {16.0F, 15.0F, 14.0F, 13.0F, 12.0F, 11.0F, 10.0F, 9.0F,
          8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
}

/**
 * @brief Computes max absolute difference between vectors.
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

/**
 * @brief Executes one CUDA-vs-CPU matmul comparison case.
 * @param n Matrix dimension.
 * @param a Matrix A in row-major order.
 * @param b Matrix B in row-major order.
 * @return True if CPU and CUDA outputs match within tolerance.
 * @sideeffects Prints max absolute difference.
 */
bool run_case(const std::size_t n, const std::vector<float>& a, const std::vector<float>& b) {
  opengpu::backends::cuda::CUDABackend backend;
  backend.set_pending_matmul_inputs(a, b, n);

  const opengpu::runtime::Kernel kernel{"matmul", 1U};
  const opengpu::runtime::LaunchConfig launch{
      opengpu::runtime::Dim3{1U, 1U, 1U},
      opengpu::runtime::Dim3{16U, 16U, 1U},
  };

  if (!backend.launch(kernel, launch)) {
    std::cerr << "CUDA backend launch failed for n=" << n << '\n';
    return false;
  }

  const std::vector<float> cuda_output = backend.last_output();
  const std::vector<float> cpu_output = opengpu::backends::cpu::kernels::matmul(a, b, n);
  const float diff = max_abs_diff(cpu_output, cuda_output);

  std::cout << std::fixed << std::setprecision(6)
            << "n=" << n << " CPU vs CUDA max abs diff: " << diff << '\n';

  if (diff > kTolerance) {
    std::cerr << "Mismatch: max diff exceeds tolerance for n=" << n << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const std::vector<float> small_a = make_small_a();
  const std::vector<float> small_b = make_small_b();
  if (!run_case(kSmallDim, small_a, small_b)) {
    return EXIT_FAILURE;
  }

  const std::vector<float> large_a = make_matrix(kLargeDim, 3U);
  const std::vector<float> large_b = make_matrix(kLargeDim, 11U);
  if (!run_case(kLargeDim, large_a, large_b)) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
