# Chapter 5 targets
add_executable(matmul_tiled matmul_tiled.cu)
add_executable(matmul_tiled_dynamic matmul_tiled_dynamic.cu)

# Link to common utilities and image I/O
target_link_libraries(matmul_tiled PRIVATE utils)
target_link_libraries(matmul_tiled_dynamic PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(matmul_tiled matmul_tiled_dynamic PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
