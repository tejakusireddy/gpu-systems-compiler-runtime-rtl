// filename: launch_config.h
// purpose: Defines launch dimensions for kernel execution.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_RUNTIME_LAUNCH_CONFIG_H_
#define OPENGPU_LAB_RUNTIME_LAUNCH_CONFIG_H_

#include <cstdint>

namespace opengpu::runtime {

/**
 * @brief Represents a 3D execution dimension.
 * @param x Size of X dimension.
 * @param y Size of Y dimension.
 * @param z Size of Z dimension.
 * @return N/A.
 * @sideeffects None.
 */
struct Dim3 {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
};

/**
 * @brief Captures grid and block dimensions for kernel launch.
 * @param grid Grid dimensions.
 * @param block Block dimensions.
 * @return N/A.
 * @sideeffects None.
 */
struct LaunchConfig {
  Dim3 grid;
  Dim3 block;
};

}  // namespace opengpu::runtime

#endif  // OPENGPU_LAB_RUNTIME_LAUNCH_CONFIG_H_
