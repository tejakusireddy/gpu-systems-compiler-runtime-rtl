// filename: test_matmul_cpu.cpp
// purpose: Validates CPU matmul correctness for Phase 1 sizes.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kSmallDim = 4U;
constexpr std::size_t kLargeDim = 64U;
constexpr float kTolerance = 1.0e-5F;

constexpr std::size_t kSquareSmallElements = kSmallDim * kSmallDim;

/**
 * @brief Prints a row-major matrix with integer-like formatting.
 * @param label Matrix label to print.
 * @param matrix Matrix contents in row-major order.
 * @param n Matrix dimension.
 * @return None.
 * @sideeffects Writes matrix values to stdout.
 */
void print_matrix(const char* label, const std::vector<float>& matrix, const std::size_t n) {
  std::cout << label << '\n';
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      std::cout << static_cast<int>(matrix[(row * n) + col]) << ' ';
    }
    std::cout << '\n';
  }
}

/**
 * @brief Builds deterministic matrix values for repeatable tests.
 * @param n Matrix dimension.
 * @param seed Seed offset for pattern generation.
 * @return Matrix data in row-major order.
 * @sideeffects None.
 */
std::vector<float> make_matrix(const std::size_t n, const std::size_t seed) {
  std::vector<float> out(n * n, 0.0F);
  constexpr std::size_t kValueMod = 23U;
  constexpr float kScale = 0.125F;

  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::size_t raw = (i + seed) % kValueMod;
    out[i] = static_cast<float>(raw) * kScale;
  }
  return out;
}

/**
 * @brief Computes golden matrix multiply output using local reference loops.
 * @param a Left matrix.
 * @param b Right matrix.
 * @param n Matrix dimension.
 * @return Golden output matrix.
 * @sideeffects None.
 */
std::vector<float> golden_matmul(const std::vector<float>& a, const std::vector<float>& b,
                                 const std::size_t n) {
  std::vector<float> c(n * n, 0.0F);
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

/**
 * @brief Validates and prints the 4x4 hardcoded matrix multiply case.
 * @param None.
 * @return True when all elements match expected values.
 * @sideeffects Writes matrices and per-element comparison results to stdout/stderr.
 */
bool run_small_case_with_prints() {
  const std::vector<float> a{
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
      9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
  const std::vector<float> b{
      16.0F, 15.0F, 14.0F, 13.0F, 12.0F, 11.0F, 10.0F, 9.0F,
      8.0F, 7.0F, 6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
  const std::vector<float> expected{
      80.0F, 70.0F, 60.0F, 50.0F, 240.0F, 214.0F, 188.0F, 162.0F,
      400.0F, 358.0F, 316.0F, 274.0F, 560.0F, 502.0F, 444.0F, 386.0F};

  const std::vector<float> actual = opengpu::backends::cpu::kernels::matmul(a, b, kSmallDim);

  print_matrix("A:", a, kSmallDim);
  print_matrix("B:", b, kSmallDim);
  print_matrix("C (computed):", actual, kSmallDim);
  print_matrix("C (expected):", expected, kSmallDim);

  float max_diff = 0.0F;
  bool all_match = true;
  for (std::size_t idx = 0; idx < kSquareSmallElements; ++idx) {
    const float diff = std::fabs(actual[idx] - expected[idx]);
    max_diff = std::max(max_diff, diff);
    const std::size_t row = idx / kSmallDim;
    const std::size_t col = idx % kSmallDim;
    const bool match = diff <= kTolerance;
    all_match = all_match && match;
    std::cout << "Match [" << row << "," << col << "]: " << (match ? "yes" : "no")
              << " (computed=" << actual[idx] << ", expected=" << expected[idx]
              << ", diff=" << diff << ")\n";
  }

  if (!all_match) {
    std::cerr << "matmul mismatch for n=" << kSmallDim << '\n';
    return false;
  }

  std::cout << std::fixed << std::setprecision(6)
            << "All elements match. Max diff: " << max_diff << '\n';
  return true;
}

/**
 * @brief Executes one generated matrix multiply correctness case.
 * @param n Matrix dimension.
 * @return True if kernel output matches computed golden output.
 * @sideeffects Writes failure details to stderr.
 */
bool run_generated_case(const std::size_t n) {
  const std::vector<float> a = make_matrix(n, 3U);
  const std::vector<float> b = make_matrix(n, 11U);
  const std::vector<float> golden = golden_matmul(a, b, n);
  const std::vector<float> actual = opengpu::backends::cpu::kernels::matmul(a, b, n);

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - golden[i]) > kTolerance) {
      std::cerr << "matmul mismatch for n=" << n << '\n';
      return false;
    }
  }
  if (actual.size() != golden.size()) {
    std::cerr << "matmul size mismatch for n=" << n << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!run_small_case_with_prints()) {
    return EXIT_FAILURE;
  }
  if (!run_generated_case(kLargeDim)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
