// filename: test_scheduler.cpp
// purpose: Validates derived scheduler metrics for healthy and bad launches.
// phase: Phase 4 - Scheduler
// last modified: 2026-04-29

#include "runtime/launch_config.h"
#include "scheduler/scheduler.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

constexpr float kPrintPrecision = 6.0F;
constexpr float kTolerance = 1.0e-6F;

/**
 * @brief Prints all scheduler metrics for a named scenario.
 * @param scenario_name Scenario title.
 * @param metrics Derived scheduler metrics.
 * @return None.
 * @sideeffects Writes formatted metrics to stdout.
 */
void print_metrics(const char* scenario_name, const opengpu::scheduler::SchedulerMetrics& metrics) {
  std::cout << "=== " << scenario_name << " ===\n";
  std::cout << "total_threads   : " << metrics.total_threads << '\n';
  std::cout << "total_warps     : " << metrics.total_warps << '\n';
  std::cout << std::fixed << std::setprecision(static_cast<int>(kPrintPrecision));
  std::cout << "occupancy       : " << metrics.occupancy << '\n';
  std::cout << "stall_fraction  : " << metrics.stall_fraction << '\n';
  std::cout << "warp_utilization: " << metrics.warp_utilization << '\n';
  std::cout << "memory_coalesced: " << (metrics.memory_coalesced ? "true" : "false") << '\n';
  std::cout << "low_occupancy   : " << (metrics.low_occupancy ? "true" : "false") << '\n';
  std::cout << "high_stall      : " << (metrics.high_stall ? "true" : "false") << "\n\n";
}

/**
 * @brief Compares two floating-point values using absolute tolerance.
 * @param lhs First value.
 * @param rhs Second value.
 * @return True if values are within tolerance.
 * @sideeffects None.
 */
bool nearly_equal(const float lhs, const float rhs) {
  return std::fabs(lhs - rhs) <= kTolerance;
}

}  // namespace

int main() {
  const opengpu::runtime::LaunchConfig scenario_a{
      opengpu::runtime::Dim3{16U, 1U, 1U},
      opengpu::runtime::Dim3{128U, 1U, 1U},
  };
  const opengpu::runtime::LaunchConfig scenario_b{
      opengpu::runtime::Dim3{1U, 1U, 1U},
      opengpu::runtime::Dim3{48U, 1U, 1U},
  };

  opengpu::scheduler::WarpScheduler scheduler_a(scenario_a);
  const opengpu::scheduler::SchedulerMetrics metrics_a = scheduler_a.simulate();
  print_metrics("Scenario A -- Healthy Launch", metrics_a);

  opengpu::scheduler::WarpScheduler scheduler_b(scenario_b);
  const opengpu::scheduler::SchedulerMetrics metrics_b = scheduler_b.simulate();
  print_metrics("Scenario B -- Bad Launch", metrics_b);

  if (metrics_a.total_threads != 2048 || metrics_a.total_warps != 64 ||
      !nearly_equal(metrics_a.occupancy, 0.500000F) ||
      !nearly_equal(metrics_a.stall_fraction, 0.150000F) ||
      !nearly_equal(metrics_a.warp_utilization, 1.000000F) ||
      !metrics_a.memory_coalesced || metrics_a.low_occupancy || metrics_a.high_stall) {
    std::cerr << "Scenario A metrics mismatch\n";
    return EXIT_FAILURE;
  }

  if (metrics_b.total_threads != 48 || metrics_b.total_warps != 2 ||
      !nearly_equal(metrics_b.occupancy, 0.015625F) ||
      !nearly_equal(metrics_b.stall_fraction, 0.004688F) ||
      !nearly_equal(metrics_b.warp_utilization, 1.000000F) ||
      metrics_b.memory_coalesced || !metrics_b.low_occupancy || metrics_b.high_stall) {
    std::cerr << "Scenario B metrics mismatch\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
