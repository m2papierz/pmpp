# Chapter 10 targets
add_executable(reduction_benchmark
    benchmark.cu
    kernels.cu
    reductions.cu
)

# Link to common utilities and image I/O
target_link_libraries(reduction_benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(reduction_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
