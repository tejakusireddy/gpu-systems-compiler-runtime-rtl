// filename: test_rtl_sim.cpp
// purpose: Validates RTL simulation backend output against CPU golden output.
// phase: Phase 6 - Integration
// last modified: 2026-04-29

#include "backends/rtl_sim/rtl_sim_backend.h"
#include "matmul.h"
#include "runtime/kernel.h"
#include "runtime/launch_config.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kDim = 4U;
constexpr std::size_t kElemCount = kDim * kDim;
constexpr float kTolerance = 1.0e-5F;

/**
 * @brief Computes max absolute difference between vectors.
 * @param lhs First vector.
 * @param rhs Second vector.
 * @return Maximum absolute element-wise difference.
 * @sideeffects None.
 */
float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
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

  opengpu::backends::rtl_sim::RTLSimBackend backend;
  backend.set_pending_matmul_inputs(a, b, kDim);

  const opengpu::runtime::Kernel kernel{"matmul", 1U};
  const opengpu::runtime::LaunchConfig launch{
      opengpu::runtime::Dim3{1U, 1U, 1U},
      opengpu::runtime::Dim3{16U, 16U, 1U},
  };
  if (!backend.launch(kernel, launch)) {
    std::cerr << "RTL sim backend launch failed\n";
    return EXIT_FAILURE;
  }

  const std::vector<float> rtl_out = backend.last_output();
  const std::vector<float> cpu_out = opengpu::backends::cpu::kernels::matmul(a, b, kDim);
  if (rtl_out.size() != kElemCount || cpu_out.size() != kElemCount) {
    std::cerr << "Unexpected output size\n";
    return EXIT_FAILURE;
  }

  for (std::size_t i = 0; i < kElemCount; ++i) {
    const float diff = std::fabs(rtl_out[i] - cpu_out[i]);
    std::cout << std::fixed << std::setprecision(6)
              << "idx=" << i << " rtl=" << rtl_out[i] << " cpu=" << cpu_out[i]
              << " diff=" << diff << '\n';
  }

  const float max_diff = max_abs_diff(rtl_out, cpu_out);
  std::cout << std::fixed << std::setprecision(6)
            << "max diff: " << max_diff << '\n';
  if (max_diff > kTolerance) {
    std::cerr << "RTL output mismatch\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
