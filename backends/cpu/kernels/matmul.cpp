// filename: matmul.cpp
// purpose: Implements naive CPU matrix multiply reference kernel.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#include "matmul.h"

#include <stdexcept>

namespace opengpu::backends::cpu::kernels {

std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b,
                          const std::size_t n) {
  const std::size_t expected_size = n * n;
  if (a.size() != expected_size || b.size() != expected_size) {
    throw std::invalid_argument("matmul input dimensions do not match n*n");
  }

  std::vector<float> c(expected_size, 0.0F);

  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      float accum = 0.0F;
      for (std::size_t k = 0; k < n; ++k) {
        accum += a[(row * n) + k] * b[(k * n) + col];
      }
      c[(row * n) + col] = accum;
    }
  }

  return c;
}

}  // namespace opengpu::backends::cpu::kernels
