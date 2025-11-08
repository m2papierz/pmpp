# Chapter 2 targets
add_executable(vec_add vec_add.cu)

# Link to common utilities
target_link_libraries(vec_add PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(vec_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
