// filename: metrics.h
// purpose: Defines profiler metric payload with scheduler-derived signals.
// phase: Phase 7 - Profiling
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_PROFILER_METRICS_H_
#define OPENGPU_LAB_PROFILER_METRICS_H_

#include <string>

namespace opengpu::profiler {

/**
 * @brief Represents latency and throughput measurements.
 * @param latency_ms End-to-end latency in milliseconds.
 * @param throughput_ops_per_sec Throughput in operations/second.
 * @return N/A.
 * @sideeffects None.
 */
struct Metrics {
  std::string backend_name;
  double latency_ms;
  double throughput_ops_per_sec;
  float occupancy;
  float stall_fraction;
  bool memory_coalesced;
  bool low_occupancy;
  bool high_stall;
};

}  // namespace opengpu::profiler

#endif  // OPENGPU_LAB_PROFILER_METRICS_H_
