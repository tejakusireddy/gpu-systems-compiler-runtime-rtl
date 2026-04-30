// filename: profiler.cpp
// purpose: Implements profiler storage and rule-based optimization insights.
// phase: Phase 7 - Profiling
// last modified: 2026-04-29

#include "profiler/profiler.h"

#include <iomanip>
#include <iostream>
#include <string>

namespace opengpu::profiler {

void Profiler::record(const Metrics& metrics) { metrics_.push_back(metrics); }

void Profiler::report(const bool compiler_coalesced) const {
  std::cout << "=== Performance Report ===\n";
  std::cout << std::left << std::setw(12) << "Backend" << std::setw(14) << "Latency(ms)"
            << std::setw(22) << "Throughput(ops/s)" << std::setw(12) << "Occupancy"
            << "Stall\n";
  std::cout << std::fixed << std::setprecision(3);
  for (const Metrics& m : metrics_) {
    std::cout << std::left << std::setw(12) << m.backend_name << std::setw(14) << m.latency_ms
              << std::setw(22) << m.throughput_ops_per_sec << std::setw(12) << m.occupancy
              << m.stall_fraction << '\n';
  }

  std::cout << "=== Optimization Insights ===\n";
  bool printed_any_insight = false;
  for (const Metrics& m : metrics_) {
    if (m.low_occupancy) {
      std::cout << "[!] low_occupancy detected on backend '" << m.backend_name
                << "' (occupancy=" << std::setprecision(2) << m.occupancy << ")\n"
                << "    -> Increase threads per block or batch size\n";
      printed_any_insight = true;
    }
    if (!m.memory_coalesced) {
      std::cout << "[!] memory not coalesced on backend '" << m.backend_name << "'\n"
                << "    -> Potential 20-30% performance loss\n";
      printed_any_insight = true;
    }
    if (m.high_stall) {
      std::cout << "[!] high warp stalls on backend '" << m.backend_name
                << "' (stall_fraction=" << std::setprecision(2) << m.stall_fraction << ")\n"
                << "    -> Likely memory latency bottleneck\n";
      printed_any_insight = true;
    }
    if (m.latency_ms > 10.0) {
      std::cout << "[!] backend '" << m.backend_name << "' latency is high (latency_ms="
                << std::setprecision(3) << m.latency_ms << ")\n"
                << "    -> Consider kernel fusion or batching\n";
      printed_any_insight = true;
    }
  }

  const Metrics* cpu = nullptr;
  const Metrics* cuda = nullptr;
  for (const Metrics& m : metrics_) {
    if (m.backend_name == "cpu") {
      cpu = &m;
    } else if (m.backend_name == "cuda") {
      cuda = &m;
    }
  }
  if (cpu != nullptr && cuda != nullptr && cuda->latency_ms > 0.0) {
    const double cuda_speedup = cpu->latency_ms / cuda->latency_ms;
    if (cuda_speedup < 2.0) {
      std::cout << "[!] CUDA underutilization detected (speedup=" << std::setprecision(3)
                << cuda_speedup << "x)\n";
      printed_any_insight = true;
    }
  }

  if (!compiler_coalesced) {
    std::cout << "[!] Compiler: memory access not coalesced\n";
    std::cout << "    -> Auto-fix pass available\n";
    std::cout << "    -> Run auto_coalescing_fix_pass() to rewrite IR\n";
    printed_any_insight = true;
  }

  if (!printed_any_insight) {
    std::cout << "[✓] All backends within expected throughput range\n";
  }
}

void Profiler::roofline_report(const std::vector<RooflineMetrics>& metrics) const {
  constexpr double kGiga = 1.0e9;
  if (metrics.empty()) {
    return;
  }

  const double peak_compute_gflops = metrics.front().peak_compute_gflops;
  const double peak_bandwidth_gbps = metrics.front().peak_bandwidth_gbps;
  const double ridge_point = metrics.front().ridge_point;

  std::cout << "=== Roofline Analysis ===\n";
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "Peak Compute  : " << peak_compute_gflops << " GFLOPS\n";
  std::cout << "Peak Bandwidth: " << peak_bandwidth_gbps << " GB/s\n";
  std::cout << "Ridge Point   : " << ridge_point << " FLOPS/byte\n\n";

  std::cout << std::left << std::setw(12) << "Backend" << std::setw(16) << "ArithInt(F/B)"
            << std::setw(20) << "Achieved(GFLOPS)" << std::setw(13) << "Bound"
            << "Assessment\n";
  std::cout << std::setprecision(2);
  for (const RooflineMetrics& m : metrics) {
    const char* bound = m.memory_bound ? "MEMORY" : "COMPUTE";
    const char* assessment =
        m.memory_bound ? "Below ridge: memory reuse helps" : "Above ridge: compute bound";
    std::cout << std::left << std::setw(12) << m.backend_name << std::setw(16)
              << m.arithmetic_intensity << std::setw(20) << (m.achieved_flops_per_sec / kGiga)
              << std::setw(13) << bound << assessment << '\n';
  }

  std::cout << "\n=== Roofline Insights ===\n";
  for (const RooflineMetrics& m : metrics) {
    if (m.arithmetic_intensity <= m.ridge_point) {
      std::cout << "[!] " << m.backend_name << " is memory-bound (intensity="
                << m.arithmetic_intensity << ", ridge=" << m.ridge_point << ")\n";
      std::cout << "    -> Increase arithmetic intensity via loop tiling\n";
      std::cout << "    -> Target: tile_size=32 increases reuse by ~32x\n";
    } else {
      std::cout << "[✓] " << m.backend_name
                << " is compute-bound — memory access pattern is not the bottleneck\n";
    }
  }
}

}  // namespace opengpu::profiler
