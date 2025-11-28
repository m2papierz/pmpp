# Chapter 11 targets
add_executable(scan_benchmark
    benchmark.cu
    kernels.cu
    scan.cu
)

# Link to common utilities and image I/O
target_link_libraries(scan_benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(scan_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
