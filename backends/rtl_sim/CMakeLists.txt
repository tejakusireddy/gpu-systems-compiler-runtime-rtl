# filename: CMakeLists.txt
# purpose: Builds RTL simulation backend placeholder.
# phase: Phase 1 - Foundation
# last modified: 2026-04-29

add_library(backend_rtl_sim STATIC
  src/rtl_sim_backend.cpp
)

target_include_directories(backend_rtl_sim
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../verilator/harness
)

target_compile_features(backend_rtl_sim PUBLIC cxx_std_17)
target_link_libraries(backend_rtl_sim PUBLIC runtime verilator_harness)
