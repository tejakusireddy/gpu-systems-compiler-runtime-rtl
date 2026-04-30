// filename: matmul.cuh
// purpose: Declares CUDA matmul kernel launcher interfaces.
// phase: Phase 2 - Real GPU
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_BACKENDS_CUDA_KERNELS_MATMUL_CUH_
#define OPENGPU_LAB_BACKENDS_CUDA_KERNELS_MATMUL_CUH_

#include <cstddef>

namespace opengpu::backends::cuda::kernels {

/**
 * @brief Launches naive CUDA matmul kernel with one thread per output element.
 * @param a_device Device pointer for matrix A (row-major, n*n).
 * @param b_device Device pointer for matrix B (row-major, n*n).
 * @param c_device Device pointer for matrix C output (row-major, n*n).
 * @param n Matrix dimension.
 * @return True on successful kernel launch and synchronization.
 * @sideeffects Executes CUDA kernel on default stream.
 */
bool launch_naive_matmul(const float* a_device, const float* b_device, float* c_device,
                         std::size_t n);

}  // namespace opengpu::backends::cuda::kernels

#endif  // OPENGPU_LAB_BACKENDS_CUDA_KERNELS_MATMUL_CUH_
