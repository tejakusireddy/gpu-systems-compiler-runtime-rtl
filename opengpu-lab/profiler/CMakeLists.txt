# filename: CMakeLists.txt
# purpose: Builds profiler components.
# phase: Phase 1 - Foundation
# last modified: 2026-04-29

add_library(profiler STATIC
  src/profiler.cpp
)

target_include_directories(profiler
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(profiler PUBLIC cxx_std_17)
