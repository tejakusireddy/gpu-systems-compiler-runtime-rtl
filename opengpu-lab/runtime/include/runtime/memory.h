// filename: memory.h
// purpose: Declares runtime memory allocation abstractions.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#ifndef OPENGPU_LAB_RUNTIME_MEMORY_H_
#define OPENGPU_LAB_RUNTIME_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <memory>

namespace opengpu::runtime {

using ByteBuffer = std::unique_ptr<std::uint8_t[]>;

/**
 * @brief Allocates zero-initialized runtime memory.
 * @param bytes Number of bytes to allocate.
 * @return Owning smart pointer for allocated memory.
 * @sideeffects Heap allocation occurs.
 */
ByteBuffer allocate_buffer(std::size_t bytes);

/**
 * @brief Releases runtime memory by resetting the owner.
 * @param buffer Owning buffer reference.
 * @return None.
 * @sideeffects Buffer ownership is released.
 */
void free_buffer(ByteBuffer& buffer);

}  // namespace opengpu::runtime

#endif  // OPENGPU_LAB_RUNTIME_MEMORY_H_
