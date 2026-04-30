// filename: passes.h
// purpose: Declares compiler optimization and analysis passes on KernelIR.
// phase: Phase 8 - Compiler
// last modified: 2026-04-30

#ifndef OPENGPU_LAB_COMPILER_PASSES_H_
#define OPENGPU_LAB_COMPILER_PASSES_H_

#include <cstddef>

#include "compiler/ir.h"

namespace opengpu::compiler {

/**
 * @brief Inserts TILE ops before each MUL op.
 * @param kernel Input kernel IR.
 * @param tile_size Tile size to annotate on inserted TILE ops.
 * @return Transformed IR with TILE operations inserted.
 * @sideeffects None.
 */
KernelIR loop_tiling_pass(const KernelIR& kernel, std::size_t tile_size);

/**
 * @brief Analyzes whether all LOAD ops are coalesced from TILE context.
 * @param kernel Input kernel IR.
 * @return True if all LOAD ops have preceding TILE op with multiple-of-32 tile.
 * @sideeffects None.
 */
bool memory_coalescing_pass(const KernelIR& kernel);

/**
 * @brief Rewrites TILE op sizes to nearest coalesced multiple of 32.
 * @param kernel Input kernel IR.
 * @return Rewritten IR with TILE sizes auto-fixed for coalescing.
 * @sideeffects None.
 */
KernelIR auto_coalescing_fix_pass(const KernelIR& kernel);

/**
 * @brief Classifies global memory op access patterns from stride annotation.
 * @param kernel Input kernel IR.
 * @return IR annotated with access_pattern for GLOBAL_LOAD/GLOBAL_STORE ops.
 * @sideeffects None.
 */
KernelIR memory_pattern_analysis_pass(const KernelIR& kernel);

}  // namespace opengpu::compiler

#endif  // OPENGPU_LAB_COMPILER_PASSES_H_
