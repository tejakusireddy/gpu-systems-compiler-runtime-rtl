// filename: passes.cpp
// purpose: Implements loop tiling and memory coalescing analysis passes.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#include "compiler/passes.h"

namespace opengpu::compiler {

KernelIR loop_tiling_pass(const KernelIR& kernel, const std::size_t tile_size) {
  KernelIR transformed{};
  transformed.name = kernel.name;

  for (const Op& op : kernel.ops) {
    if (op.type == OpType::MUL) {
      transformed.ops.push_back(
          Op{OpType::TILE, "tile", "", "", tile_size, MemAccessPattern::UNKNOWN, 0U});
    }
    transformed.ops.push_back(op);
  }
  return transformed;
}

bool memory_coalescing_pass(const KernelIR& kernel) {
  bool saw_load = false;
  bool saw_valid_tiling_context = false;

  for (const Op& op : kernel.ops) {
    if (op.type == OpType::LOAD) {
      saw_load = true;
      continue;
    }
    if (op.type == OpType::TILE) {
      if ((op.tile_size % 32U) != 0U) {
        return false;
      }
      saw_valid_tiling_context = true;
    }
  }
  return saw_load ? saw_valid_tiling_context : true;
}

KernelIR auto_coalescing_fix_pass(const KernelIR& kernel) {
  KernelIR rewritten = kernel;
  for (Op& op : rewritten.ops) {
    if (op.type == OpType::TILE && (op.tile_size % 32U) != 0U) {
      op.tile_size = ((op.tile_size + 31U) / 32U) * 32U;
    }
  }
  return rewritten;
}

KernelIR memory_pattern_analysis_pass(const KernelIR& kernel) {
  KernelIR annotated = kernel;
  for (Op& op : annotated.ops) {
    if (op.type == OpType::GLOBAL_LOAD || op.type == OpType::GLOBAL_STORE) {
      if (op.stride == 1U) {
        op.access_pattern = MemAccessPattern::COALESCED;
      } else if (op.stride > 1U) {
        op.access_pattern = MemAccessPattern::STRIDED;
      } else {
        op.access_pattern = MemAccessPattern::UNKNOWN;
      }
    }
  }
  return annotated;
}

}  // namespace opengpu::compiler
