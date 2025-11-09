# Chapter 6 targets
add_executable(matmul_thread_coarsening matmul_thread_coarsening.cu)
add_executable(matmul_corner_turning matmul_corner_turning.cu)

# Link to common utilities and image I/O
target_link_libraries(matmul_thread_coarsening PRIVATE utils)
target_link_libraries(matmul_corner_turning PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(matmul_thread_coarsening matmul_corner_turning PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
