# filename: CMakeLists.txt
# purpose: Builds CUDA backend and naive CUDA matmul kernel target.
# phase: Phase 2 - Real GPU
# last modified: 2026-04-29

add_library(backend_cuda STATIC
  src/cuda_backend.cu
  kernels/matmul.cu
)

set_source_files_properties(src/cuda_backend.cu kernels/matmul.cu PROPERTIES LANGUAGE CXX)

target_include_directories(backend_cuda
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/include
)

target_compile_features(backend_cuda PUBLIC cxx_std_17)
