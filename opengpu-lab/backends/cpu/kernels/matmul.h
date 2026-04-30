// filename: matmul.h
// purpose: Declares CPU reference matrix multiply kernel.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_BACKENDS_CPU_KERNELS_MATMUL_H_
#define OPENGPU_LAB_BACKENDS_CPU_KERNELS_MATMUL_H_

#include <cstddef>
#include <vector>

namespace opengpu::backends::cpu::kernels {

/**
 * @brief Computes C = A x B for row-major square matrices.
 * @param a Left input matrix data (size n*n).
 * @param b Right input matrix data (size n*n).
 * @param n Matrix dimension.
 * @return Result matrix C of size n*n.
 * @sideeffects None.
 */
std::vector<float> matmul(const std::vector<float>& a, const std::vector<float>& b,
                          std::size_t n);

}  // namespace opengpu::backends::cpu::kernels

#endif  // OPENGPU_LAB_BACKENDS_CPU_KERNELS_MATMUL_H_
