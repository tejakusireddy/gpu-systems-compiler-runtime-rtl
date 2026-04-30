// filename: cpu_backend.cpp
// purpose: Implements CPU backend launch and matmul execution path.
// phase: Phase 3 - Runtime Abstraction
// last modified: 2026-04-29

#include "backends/cpu/cpu_backend.h"

#include <utility>

#include "matmul.h"

namespace opengpu::backends::cpu {

bool CPUBackend::launch(const opengpu::runtime::Kernel& kernel,
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

  last_output_ = kernels::matmul(pending_a_, pending_b_, pending_n_);
  return true;
}

void CPUBackend::set_pending_matmul_inputs(const std::vector<float>& a,
                                           const std::vector<float>& b,
                                           const std::size_t n) {
  pending_a_ = a;
  pending_b_ = b;
  pending_n_ = n;
}

std::vector<float> CPUBackend::last_output() const { return last_output_; }

}  // namespace opengpu::backends::cpu
