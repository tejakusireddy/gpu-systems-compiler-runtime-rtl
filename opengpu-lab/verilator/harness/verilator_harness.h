// filename: verilator_harness.h
// purpose: Declares Verilator harness for RTL matmul execution.
// phase: Phase 6 - Integration
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_VERILATOR_HARNESS_H_
#define OPENGPU_LAB_VERILATOR_HARNESS_H_

#include <cstddef>
#include <vector>

namespace opengpu::verilator {

/**
 * @brief Placeholder harness for RTL simulation orchestration.
 * @param None.
 * @return N/A.
 * @sideeffects None in Phase 1.
 */
class VerilatorHarness {
 public:
  VerilatorHarness() = default;

  /**
   * @brief Runs matrix multiply by driving the Verilated RTL model.
   * @param a Input matrix A in row-major order.
   * @param b Input matrix B in row-major order.
   * @param n Matrix dimension (currently requires 4).
   * @return Output matrix C in row-major order.
   * @sideeffects Executes clocked simulation cycles in Verilated model.
   */
  std::vector<float> run_matmul(const std::vector<float>& a, const std::vector<float>& b,
                                std::size_t n) const;
};

}  // namespace opengpu::verilator

#endif  // OPENGPU_LAB_VERILATOR_HARNESS_H_
