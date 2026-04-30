# filename: CMakeLists.txt
# purpose: Builds scheduler simulation components.
# phase: Phase 1 - Foundation
# last modified: 2026-04-29

add_library(scheduler STATIC
  src/warp.cpp
  src/sm.cpp
  src/scheduler.cpp
)

target_include_directories(scheduler
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
)

target_compile_features(scheduler PUBLIC cxx_std_17)
