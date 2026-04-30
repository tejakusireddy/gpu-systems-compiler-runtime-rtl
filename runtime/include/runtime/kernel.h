// filename: kernel.h
// purpose: Defines generic kernel metadata used by runtime/backends.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_RUNTIME_KERNEL_H_
#define OPENGPU_LAB_RUNTIME_KERNEL_H_

#include <cstdint>
#include <string>

namespace opengpu::runtime {

/**
 * @brief Describes an executable kernel entity.
 * @param name Human-readable kernel identifier.
 * @param op_id Numeric operation identifier reserved for dispatch routing.
 * @return N/A.
 * @sideeffects None.
 */
struct Kernel {
  std::string name;
  std::uint32_t op_id;
};

}  // namespace opengpu::runtime

#endif  // OPENGPU_LAB_RUNTIME_KERNEL_H_
