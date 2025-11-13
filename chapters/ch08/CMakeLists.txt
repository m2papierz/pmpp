# Chapter 8 targets
add_executable(stencil_benchmark
    stencil_sweep.cu
    benchmark.cu
)

# Link to common utilities and image I/O
target_link_libraries(stencil_benchmark PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(stencil_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
