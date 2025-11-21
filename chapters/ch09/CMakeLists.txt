# Chapter 9 targets
add_executable(hist_benchmark
    benchmark.cu
    kernels.cu
    histogram.cu
)

# Link to common utilities and image I/O
target_link_libraries(hist_benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(hist_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
