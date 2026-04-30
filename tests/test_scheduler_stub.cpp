// filename: test_scheduler_stub.cpp
// purpose: Scheduler test placeholder.
// phase: Phase 4 - Scheduler
// last modified: 2026-04-29

#include "runtime/launch_config.h"
#include "scheduler/scheduler.h"

#include <cstdlib>

int main() {
  const opengpu::runtime::LaunchConfig launch{
      opengpu::runtime::Dim3{1U, 1U, 1U},
      opengpu::runtime::Dim3{32U, 1U, 1U},
  };
  opengpu::scheduler::WarpScheduler scheduler(launch);
  const opengpu::scheduler::SchedulerMetrics metrics = scheduler.simulate();
  // TODO(phase-4): Remove stub once full scheduler coverage expands.
  return (metrics.total_warps >= 1) ? EXIT_SUCCESS : EXIT_FAILURE;
}
