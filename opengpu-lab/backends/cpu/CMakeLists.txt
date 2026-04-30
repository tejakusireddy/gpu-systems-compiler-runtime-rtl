# filename: CMakeLists.txt
# purpose: Builds CPU backend and CPU kernels.
# phase: Phase 1 - Foundation
# last modified: 2026-04-29

add_library(backend_cpu STATIC
  src/cpu_backend.cpp
  kernels/matmul.cpp
)

target_include_directories(backend_cpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include
)

target_compile_features(backend_cpu PUBLIC cxx_std_17)

target_link_libraries(backend_cpu PUBLIC runtime)
