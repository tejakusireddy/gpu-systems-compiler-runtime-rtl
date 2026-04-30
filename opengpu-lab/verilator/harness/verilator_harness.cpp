// filename: verilator_harness.cpp
// purpose: Implements Verilator harness for RTL matmul execution.
// phase: Phase 6 - Integration
// last modified: 2026-04-29

#include "verilator_harness.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "Vmatmul_accelerator.h"
#include "verilated.h"

double sc_time_stamp() { return 0.0; }

namespace opengpu::verilator {

namespace {

constexpr std::size_t kModelDim = 4U;
constexpr std::size_t kElemCount = kModelDim * kModelDim;
constexpr std::size_t kMaxSimulationCycles = 2048U;

/**
 * @brief Advances the Verilated model by one full clock cycle.
 * @param model Verilated top module instance.
 * @return None.
 * @sideeffects Updates RTL state and combinational outputs.
 */
void tick(Vmatmul_accelerator* model) {
  model->clk = 0U;
  model->eval();
  model->clk = 1U;
  model->eval();
}

/**
 * @brief Executes one 4x4 matrix multiply on the Verilated model.
 * @param a_tile Row-major 4x4 tile A.
 * @param b_tile Row-major 4x4 tile B.
 * @return Row-major 4x4 output tile C.
 * @sideeffects Drives and advances the Verilated model.
 */
std::vector<float> run_tile_4x4(const std::vector<float>& a_tile,
                                const std::vector<float>& b_tile) {
  std::unique_ptr<Vmatmul_accelerator> model = std::make_unique<Vmatmul_accelerator>();
  model->start = 0U;
  model->rst = 1U;
  tick(model.get());
  tick(model.get());
  model->rst = 0U;

  for (std::size_t idx = 0; idx < kElemCount; ++idx) {
    model->a_flat[idx] = static_cast<std::uint32_t>(a_tile[idx]);
    model->b_flat[idx] = static_cast<std::uint32_t>(b_tile[idx]);
  }

  model->start = 1U;
  tick(model.get());
  model->start = 0U;

  for (std::size_t cycle = 0; cycle < kMaxSimulationCycles; ++cycle) {
    tick(model.get());
    if (model->done != 0U) {
      std::vector<float> output(kElemCount, 0.0F);
      for (std::size_t idx = 0; idx < kElemCount; ++idx) {
        output[idx] = static_cast<float>(model->c_flat[idx]);
      }
      return output;
    }
  }
  return {};
}

}  // namespace

std::vector<float> VerilatorHarness::run_matmul(const std::vector<float>& a,
                                                const std::vector<float>& b,
                                                const std::size_t n) const {
  if (n == 0U || (n % kModelDim) != 0U) {
    return {};
  }
  const std::size_t expected = n * n;
  if (a.size() != expected || b.size() != expected) {
    return {};
  }

  std::vector<float> output(expected, 0.0F);
  const std::size_t tile_count = n / kModelDim;

  for (std::size_t tile_row = 0; tile_row < tile_count; ++tile_row) {
    for (std::size_t tile_col = 0; tile_col < tile_count; ++tile_col) {
      std::vector<float> accum_tile(kElemCount, 0.0F);

      for (std::size_t tile_k = 0; tile_k < tile_count; ++tile_k) {
        std::vector<float> a_tile(kElemCount, 0.0F);
        std::vector<float> b_tile(kElemCount, 0.0F);

        for (std::size_t i = 0; i < kModelDim; ++i) {
          for (std::size_t j = 0; j < kModelDim; ++j) {
            const std::size_t a_row = (tile_row * kModelDim) + i;
            const std::size_t a_col = (tile_k * kModelDim) + j;
            const std::size_t b_row = (tile_k * kModelDim) + i;
            const std::size_t b_col = (tile_col * kModelDim) + j;
            a_tile[(i * kModelDim) + j] = a[(a_row * n) + a_col];
            b_tile[(i * kModelDim) + j] = b[(b_row * n) + b_col];
          }
        }

        const std::vector<float> tile_out = run_tile_4x4(a_tile, b_tile);
        if (tile_out.size() != kElemCount) {
          return {};
        }
        for (std::size_t idx = 0; idx < kElemCount; ++idx) {
          accum_tile[idx] += tile_out[idx];
        }
      }

      for (std::size_t i = 0; i < kModelDim; ++i) {
        for (std::size_t j = 0; j < kModelDim; ++j) {
          const std::size_t out_row = (tile_row * kModelDim) + i;
          const std::size_t out_col = (tile_col * kModelDim) + j;
          output[(out_row * n) + out_col] = accum_tile[(i * kModelDim) + j];
        }
      }
    }
  }
  return output;
}

}  // namespace opengpu::verilator
