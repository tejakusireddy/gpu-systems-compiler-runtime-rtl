// filename: dispatcher.h
// purpose: Declares runtime dispatcher for named backend routing.
// phase: Phase 3 - Runtime Abstraction
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_RUNTIME_DISPATCHER_H_
#define OPENGPU_LAB_RUNTIME_DISPATCHER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "runtime/backend.h"

namespace opengpu::runtime {

/**
 * @brief Dispatches kernels to registered named backend implementations.
 * @param None.
 * @return N/A.
 * @sideeffects Owns backend lifetime for dispatch operations.
 */
class Dispatcher {
 public:
  Dispatcher() = default;

  /**
   * @brief Registers a backend implementation under a logical backend name.
   * @param backend_name Logical backend selector, e.g. "cpu" or "cuda".
   * @param backend Shared backend implementation.
   * @return True when registration succeeds.
   * @sideeffects Replaces any backend previously registered to the same name.
   */
  bool register_backend(const std::string& backend_name, std::shared_ptr<Backend> backend);

  /**
   * @brief Routes a kernel launch to a named backend.
   * @param backend_name Logical backend selector, e.g. "cpu" or "cuda".
   * @param kernel Kernel descriptor.
   * @param launch Launch configuration.
   * @return True if launch succeeded.
   * @sideeffects May trigger backend execution.
   */
  bool dispatch(const std::string& backend_name, const Kernel& kernel,
                const LaunchConfig& launch) const;

 private:
  std::unordered_map<std::string, std::shared_ptr<Backend>> backends_;
};

}  // namespace opengpu::runtime

#endif  // OPENGPU_LAB_RUNTIME_DISPATCHER_H_
