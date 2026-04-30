# filename: CMakeLists.txt
# purpose: Builds Verilator model and C++ harness integration library.
# phase: Phase 6 - Integration
# last modified: 2026-04-29

cmake_policy(SET CMP0144 NEW)

find_package(verilator HINTS $ENV{VERILATOR_ROOT} REQUIRED)

add_library(rtl_sim_lib STATIC)

verilate(rtl_sim_lib
  SOURCES ${CMAKE_SOURCE_DIR}/rtl/matmul_accelerator.v
  PREFIX Vmatmul_accelerator
)
target_compile_options(rtl_sim_lib PRIVATE -w)

add_library(verilator_harness STATIC
  harness/verilator_harness.cpp
)

target_include_directories(verilator_harness
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/harness
)

target_compile_features(verilator_harness PUBLIC cxx_std_17)
target_compile_options(verilator_harness PRIVATE -w)
target_link_libraries(verilator_harness PUBLIC rtl_sim_lib)
