// filename: scheduler.h
// purpose: Declares warp scheduler simulation and derived metric outputs.
// phase: Phase 4 - Scheduler
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_SCHEDULER_SCHEDULER_H_
#define OPENGPU_LAB_SCHEDULER_SCHEDULER_H_

#include <vector>

#include "runtime/launch_config.h"
#include "scheduler/sm.h"
#include "scheduler/warp.h"

namespace opengpu::scheduler {

struct SchedulerMetrics {
  int total_warps;
  int total_threads;
  float occupancy;
  float stall_fraction;
  float warp_utilization;
  bool memory_coalesced;
  bool low_occupancy;
  bool high_stall;
};

/**
 * @brief Simulates warp scheduling across SMs and computes derived metrics.
 * @param launch Launch configuration (grid/block dimensions).
 * @return N/A.
 * @sideeffects Updates internal warp and SM simulation state.
 */
class WarpScheduler {
 public:
  explicit WarpScheduler(const opengpu::runtime::LaunchConfig& launch);

  /**
   * @brief Runs fixed-cycle scheduling simulation and returns derived metrics.
   * @param None.
   * @return SchedulerMetrics derived from simulated launch behavior.
   * @sideeffects Internal simulation state is refreshed and consumed.
   */
  SchedulerMetrics simulate();

 private:
  /**
   * @brief Creates warps from launch thread count using 32-thread warps.
   * @param None.
   * @return None.
   * @sideeffects Populates internal warp list.
   */
  void form_warps();

  /**
   * @brief Distributes formed warps round-robin across all SMs.
   * @param None.
   * @return None.
   * @sideeffects Populates SM active warp lists.
   */
  void distribute_warps();

  opengpu::runtime::LaunchConfig launch_;
  std::vector<Warp> warps_;
  std::vector<SM> sms_;
};

}  // namespace opengpu::scheduler

#endif  // OPENGPU_LAB_SCHEDULER_SCHEDULER_H_
