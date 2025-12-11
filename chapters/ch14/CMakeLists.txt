# Chapter 12 targets
add_executable(sparse_benchmark
    benchmark.cu
    kernels.cu
    formats.cu
    sparse.cu
)

# Link to common utilities and image I/O
target_link_libraries(sparse_benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(sparse_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
