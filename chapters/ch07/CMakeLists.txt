# Chapter 7 targets
add_executable(benchmark
    benchmark.cu
    kernels/kernels2d.cu
    kernels/kernels3d.cu
    functions.cu
)

# Link to common utilities and image I/O
target_link_libraries(benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
