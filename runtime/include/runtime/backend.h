// filename: backend.h
// purpose: Declares abstract backend interface for kernel execution.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_RUNTIME_BACKEND_H_
#define OPENGPU_LAB_RUNTIME_BACKEND_H_

#include "runtime/kernel.h"
#include "runtime/launch_config.h"

namespace opengpu::runtime {

/**
 * @brief Abstract execution backend contract.
 * @param kernel Kernel descriptor.
 * @param launch Launch dimensions.
 * @return True if backend accepted launch, false otherwise.
 * @sideeffects Backend-specific launch side effects may occur.
 */
class Backend {
 public:
  Backend() = default;
  Backend(const Backend&) = default;
  Backend& operator=(const Backend&) = default;
  Backend(Backend&&) noexcept = default;
  Backend& operator=(Backend&&) noexcept = default;
  virtual ~Backend() = default;

  /**
   * @brief Launches a kernel through the concrete backend implementation.
   * @param kernel Kernel metadata.
   * @param launch Grid/block launch config.
   * @return True on successful dispatch.
   * @sideeffects May enqueue or execute backend work.
   */
  virtual bool launch(const Kernel& kernel, const LaunchConfig& launch) = 0;
};

}  // namespace opengpu::runtime

#endif  // OPENGPU_LAB_RUNTIME_BACKEND_H_
