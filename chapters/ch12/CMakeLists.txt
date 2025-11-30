# Chapter 12 targets
add_executable(merge_benchmark
    benchmark.cu
    kernels.cu
    merge.cu
)

# Link to common utilities and image I/O
target_link_libraries(merge_benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(merge_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
