// filename: test_roofline.cpp
// purpose: Validates roofline model computation and backend bound classification.
// phase: Profiler v2
// last modified: 2026-04-30

#include "backends/cuda/cuda_backend.h"
#include "backends/rtl_sim/rtl_sim_backend.h"
#include "matmul.h"
#include "profiler/profiler.h"
#include "runtime/kernel.h"
#include "runtime/launch_config.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kDim = 64U;
constexpr std::size_t kElemCount = kDim * kDim;
constexpr float kTolerance = 1.0e-5F;
constexpr double kPeakComputeGflops = 10000.0;
constexpr double kPeakBandwidthGbps = 900.0;

std::vector<float> make_matrix(const std::size_t n, const std::size_t seed) {
  std::vector<float> out(n * n, 0.0F);
  constexpr std::size_t kValueMod = 23U;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<float>((i + seed) % kValueMod);
  }
  return out;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = std::fabs(lhs[i] - rhs[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

opengpu::profiler::RooflineMetrics make_roofline_metrics(const std::string& backend_name,
                                                         const double latency_ms) {
  const double n = static_cast<double>(kDim);
  const double flops = 2.0 * n * n * n;
  const double bytes = (2.0 * n * n * 4.0) + (n * n * 4.0);
  const double arithmetic_intensity = flops / bytes;
  const double latency_sec = latency_ms / 1000.0;
  const double achieved_flops = (latency_sec > 0.0) ? (flops / latency_sec) : 0.0;
  const double ridge_point = kPeakComputeGflops / kPeakBandwidthGbps;
  const bool compute_bound = arithmetic_intensity > ridge_point;
  const bool memory_bound = arithmetic_intensity <= ridge_point;
  return opengpu::profiler::RooflineMetrics{
      backend_name,        flops,          bytes,          arithmetic_intensity, achieved_flops,
      kPeakComputeGflops,  kPeakBandwidthGbps, ridge_point,   compute_bound,      memory_bound};
}

}  // namespace

int main() {
  const std::vector<float> a = make_matrix(kDim, 3U);
  const std::vector<float> b = make_matrix(kDim, 11U);

  const auto cpu_begin = std::chrono::steady_clock::now();
  const std::vector<float> cpu_out = opengpu::backends::cpu::kernels::matmul(a, b, kDim);
  const auto cpu_end = std::chrono::steady_clock::now();
  const double cpu_latency_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

  opengpu::backends::cuda::CUDABackend cuda_backend;
  const auto cuda_begin = std::chrono::steady_clock::now();
  std::vector<float> cuda_out;
  const bool cuda_ok = cuda_backend.run_matmul(a, b, kDim, &cuda_out);
  const auto cuda_end = std::chrono::steady_clock::now();
  const double cuda_latency_ms =
      std::chrono::duration<double, std::milli>(cuda_end - cuda_begin).count();
  if (!cuda_ok || cuda_out.size() != kElemCount) {
    std::cerr << "CUDA backend matmul failed\n";
    return EXIT_FAILURE;
  }

  opengpu::backends::rtl_sim::RTLSimBackend rtl_backend;
  rtl_backend.set_pending_matmul_inputs(a, b, kDim);
  const opengpu::runtime::Kernel kernel{"matmul", 1U};
  const opengpu::runtime::LaunchConfig launch{
      opengpu::runtime::Dim3{1U, 1U, 1U},
      opengpu::runtime::Dim3{16U, 16U, 1U},
  };
  const auto rtl_begin = std::chrono::steady_clock::now();
  const bool rtl_ok = rtl_backend.launch(kernel, launch);
  const auto rtl_end = std::chrono::steady_clock::now();
  const double rtl_latency_ms =
      std::chrono::duration<double, std::milli>(rtl_end - rtl_begin).count();
  const std::vector<float> rtl_out = rtl_backend.last_output();
  if (!rtl_ok || rtl_out.size() != kElemCount) {
    std::cerr << "RTL sim backend matmul failed\n";
    return EXIT_FAILURE;
  }

  if (max_abs_diff(cpu_out, cuda_out) > kTolerance || max_abs_diff(cpu_out, rtl_out) > kTolerance) {
    std::cerr << "Backend outputs diverged\n";
    return EXIT_FAILURE;
  }

  const opengpu::profiler::RooflineMetrics cpu_roof = make_roofline_metrics("cpu", cpu_latency_ms);
  const opengpu::profiler::RooflineMetrics cuda_roof =
      make_roofline_metrics("cuda", cuda_latency_ms);
  const opengpu::profiler::RooflineMetrics rtl_roof = make_roofline_metrics("rtl_sim", rtl_latency_ms);
  const std::vector<opengpu::profiler::RooflineMetrics> roofline{
      cpu_roof, cuda_roof, rtl_roof};

  for (const opengpu::profiler::RooflineMetrics& m : roofline) {
    if (m.arithmetic_intensity < 10.0 || m.arithmetic_intensity > 11.0) {
      std::cerr << "Unexpected arithmetic intensity for " << m.backend_name << '\n';
      return EXIT_FAILURE;
    }
    if (!m.memory_bound || m.compute_bound) {
      std::cerr << "Expected memory-bound classification for " << m.backend_name << '\n';
      return EXIT_FAILURE;
    }
  }

  opengpu::profiler::Profiler profiler;
  std::cout << std::fixed << std::setprecision(2);
  profiler.roofline_report(roofline);
  return EXIT_SUCCESS;
}
