// filename: warp.h
// purpose: Defines warp model used by scheduler simulation.
// phase: Phase 4 - Scheduler
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_SCHEDULER_WARP_H_
#define OPENGPU_LAB_SCHEDULER_WARP_H_

namespace opengpu::scheduler {

enum class WarpState {
  READY = 0,
  STALLED = 1,
  COMPLETE = 2,
};

/**
 * @brief Represents a warp for scheduler simulation.
 * @param warp_id Warp identifier.
 * @param thread_count Number of logical threads in warp.
 * @param state Current scheduler-visible state.
 * @return N/A.
 * @sideeffects None.
 */
struct Warp {
  int warp_id;
  int thread_count;
  WarpState state;
};

}  // namespace opengpu::scheduler

#endif  // OPENGPU_LAB_SCHEDULER_WARP_H_
