# filename: CMakeLists.txt
# purpose: Build runtime library and public interfaces.
# phase: Phase 1 - Foundation
# last modified: 2026-04-29

add_library(runtime STATIC
  src/dispatcher.cpp
  src/memory.cpp
)

target_include_directories(runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(runtime PUBLIC cxx_std_17)
