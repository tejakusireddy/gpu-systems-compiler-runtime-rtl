// filename: memory.cpp
// purpose: Implements runtime memory allocation functions.
// phase: Phase 1 - Foundation
// last modified: 2026-04-29

#include "runtime/memory.h"

#include <algorithm>

namespace opengpu::runtime {

ByteBuffer allocate_buffer(const std::size_t bytes) {
  ByteBuffer buffer = std::make_unique<std::uint8_t[]>(bytes);
  constexpr std::uint8_t kZeroFillValue = 0U;
  std::fill_n(buffer.get(), bytes, kZeroFillValue);
  return buffer;
}

void free_buffer(ByteBuffer& buffer) { buffer.reset(); }

}  // namespace opengpu::runtime
