# Chapter 6 targets
add_executable(main
    main.cu
    kernels/kernels2d.cu
    kernels/kernels3d.cu
    functions.cu
)

# Link to common utilities and image I/O
target_link_libraries(main PRIVATE utils)

# Per-target CUDA settings (inherit globals from root)
set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
