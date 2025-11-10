#include "kernels.cuh"
#include "functions.cuh"
#include "utils.hpp"

void conv2d(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    // Allocate device memory
    std::size_t arraySize { static_cast<std::size_t>(height * width * sizeof(float)) };
    std::size_t filterSize { static_cast<std::size_t>((2 * radius + 1) * sizeof(float)) };
    
    float *inArray_d { nullptr }, *filter_d { nullptr }, *outArray_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inArray_d, arraySize));
    CUDA_CHECK(cudaMalloc(&filter_d, filterSize));
    CUDA_CHECK(cudaMalloc(&outArray_d, arraySize));

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(inArray_d, inArray, arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(filter_d, filter, filterSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(outArray_d, outArray, arraySize, cudaMemcpyHostToDevice));

    // Call kernel
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
        static_cast<int>(utils::cdiv(width, BLOCK_SIZE)),
        static_cast<int>(utils::cdiv(height, BLOCK_SIZE)),
        1
    );
    conv2dKernel<<<gridSize, blockSize>>>(inArray_d, filter_d, outArray_d, radius, height, width);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(outArray, outArray_d, arraySize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(inArray_d));
    CUDA_CHECK(cudaFree(filter_d));
    CUDA_CHECK(cudaFree(outArray_d));
}
