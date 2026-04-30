// filename: sm.h
// purpose: Defines streaming multiprocessor model for scheduler simulation.
// phase: Phase 4 - Scheduler
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_SCHEDULER_SM_H_
#define OPENGPU_LAB_SCHEDULER_SM_H_

#include <vector>

#include "scheduler/warp.h"

namespace opengpu::scheduler {

constexpr int kMaxWarpSlotsPerSM = 32;

/**
 * @brief Represents a streaming multiprocessor resource view.
 * @param sm_id SM identifier.
 * @param max_warp_slots Maximum resident warp slots.
 * @param active_warps Active warp list resident on this SM.
 * @return N/A.
 * @sideeffects None.
 */
struct SM {
  int sm_id;
  const int max_warp_slots = kMaxWarpSlotsPerSM;
  std::vector<Warp> active_warps;

  /**
   * @brief Returns current occupancy as active warps over max warp slots.
   * @param None.
   * @return Occupancy ratio for this SM.
   * @sideeffects None.
   */
  float occupancy() const {
    return static_cast<float>(active_warps.size()) /
           static_cast<float>(max_warp_slots);
  }
};

}  // namespace opengpu::scheduler

#endif  // OPENGPU_LAB_SCHEDULER_SM_H_
