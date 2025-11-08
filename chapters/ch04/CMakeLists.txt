# Chapter 4 targets
add_executable(device_properties device_properties.cu)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(device_properties PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
