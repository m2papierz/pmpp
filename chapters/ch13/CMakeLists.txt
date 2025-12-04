# Chapter 12 targets
add_executable(sorting_benchmark
    benchmark.cu
    kernels/radix.cu
    kernels/merge.cu
    sorting.cu
)

# Link to common utilities and image I/O
target_link_libraries(sorting_benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(sorting_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
