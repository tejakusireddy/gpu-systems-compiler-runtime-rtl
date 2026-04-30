// filename: scheduler.cpp
// purpose: Implements warp and SM scheduling simulation with derived metrics.
// phase: Phase 4 - Scheduler
// last modified: 2026-04-29

#include "scheduler/scheduler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace opengpu::scheduler {

namespace {

constexpr int kWarpSize = 32;
constexpr int kSMCount = 4;
constexpr int kSimulationCycles = 100;
constexpr float kMaxStallFraction = 0.5F;
constexpr float kStallScaleFactor = 0.3F;
constexpr float kLowOccupancyThreshold = 0.50F;
constexpr float kHighStallThreshold = 0.25F;
constexpr float kMaxOccupancy = 1.0F;

}  // namespace

WarpScheduler::WarpScheduler(const opengpu::runtime::LaunchConfig& launch)
    : launch_(launch) {
  sms_.reserve(static_cast<std::size_t>(kSMCount));
  for (int sm_index = 0; sm_index < kSMCount; ++sm_index) {
    sms_.push_back(SM{sm_index, kMaxWarpSlotsPerSM, {}});
  }
}

void WarpScheduler::form_warps() {
  warps_.clear();
  const std::uint64_t total_threads_64 =
      static_cast<std::uint64_t>(launch_.grid.x) * launch_.grid.y * launch_.grid.z *
      launch_.block.x * launch_.block.y * launch_.block.z;
  const int total_threads = static_cast<int>(total_threads_64);
  const int total_warps = (total_threads + (kWarpSize - 1)) / kWarpSize;

  warps_.reserve(static_cast<std::size_t>(total_warps));
  for (int warp_id = 0; warp_id < total_warps; ++warp_id) {
    warps_.push_back(Warp{warp_id, kWarpSize, WarpState::READY});
  }
}

void WarpScheduler::distribute_warps() {
  for (SM& sm : sms_) {
    sm.active_warps.clear();
  }

  for (std::size_t warp_index = 0; warp_index < warps_.size(); ++warp_index) {
    const std::size_t sm_index = warp_index % sms_.size();
    sms_[sm_index].active_warps.push_back(warps_[warp_index]);
  }
}

SchedulerMetrics WarpScheduler::simulate() {
  form_warps();
  distribute_warps();

  int completed_warps = 0;
  std::size_t rr_index = 0U;
  for (int cycle = 0; cycle < kSimulationCycles; ++cycle) {
    bool progressed = false;
    for (int probe = 0; probe < kSMCount; ++probe) {
      const std::size_t sm_index = (rr_index + static_cast<std::size_t>(probe)) % sms_.size();
      SM& sm = sms_[sm_index];
      for (Warp& warp : sm.active_warps) {
        if (warp.state == WarpState::READY) {
          warp.state = WarpState::COMPLETE;
          ++completed_warps;
          progressed = true;
          break;
        }
      }
      if (progressed) {
        rr_index = (sm_index + 1U) % sms_.size();
        break;
      }
    }
    if (!progressed) {
      break;
    }
  }

  const int total_warps = static_cast<int>(warps_.size());
  const std::uint64_t total_threads_64 =
      static_cast<std::uint64_t>(launch_.grid.x) * launch_.grid.y * launch_.grid.z *
      launch_.block.x * launch_.block.y * launch_.block.z;
  const int total_threads = static_cast<int>(total_threads_64);

  const float warps_per_sm = static_cast<float>(total_warps) / static_cast<float>(kSMCount);
  const float stall_fraction = std::min(
      kMaxStallFraction, (warps_per_sm / static_cast<float>(kMaxWarpSlotsPerSM)) * kStallScaleFactor);
  const float occupancy = std::min(
      kMaxOccupancy, static_cast<float>(total_warps) /
                         (static_cast<float>(kSMCount) * static_cast<float>(kMaxWarpSlotsPerSM)));
  const float warp_utilization = (total_warps == 0)
                                     ? 0.0F
                                     : static_cast<float>(completed_warps) /
                                           static_cast<float>(total_warps);

  SchedulerMetrics metrics{};
  metrics.total_warps = total_warps;
  metrics.total_threads = total_threads;
  metrics.occupancy = occupancy;
  metrics.stall_fraction = stall_fraction;
  metrics.warp_utilization = warp_utilization;
  metrics.memory_coalesced = (launch_.block.x % static_cast<std::uint32_t>(kWarpSize) == 0U);
  metrics.low_occupancy = (metrics.occupancy < kLowOccupancyThreshold);
  metrics.high_stall = (metrics.stall_fraction > kHighStallThreshold);
  return metrics;
}

}  // namespace opengpu::scheduler
