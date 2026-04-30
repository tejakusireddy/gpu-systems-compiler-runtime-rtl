// filename: rtl_sim_backend.cpp
// purpose: Implements RTL simulation backend using Verilator harness.
// phase: Phase 6 - Integration
// last modified: 2026-04-29

#include "backends/rtl_sim/rtl_sim_backend.h"

#include <utility>
#include <vector>

#include "verilator_harness.h"

namespace opengpu::backends::rtl_sim {

bool RTLSimBackend::launch(const opengpu::runtime::Kernel& kernel,
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
  const std::size_t expected_elements = pending_n_ * pending_n_;
  if (pending_a_.size() != expected_elements || pending_b_.size() != expected_elements) {
    return false;
  }

  opengpu::verilator::VerilatorHarness harness;
  last_output_ = harness.run_matmul(pending_a_, pending_b_, pending_n_);
  return last_output_.size() == expected_elements;
}

void RTLSimBackend::set_pending_matmul_inputs(const std::vector<float>& a,
                                              const std::vector<float>& b,
                                              const std::size_t n) {
  pending_a_ = a;
  pending_b_ = b;
  pending_n_ = n;
}

std::vector<float> RTLSimBackend::last_output() const { return last_output_; }

}  // namespace opengpu::backends::rtl_sim
