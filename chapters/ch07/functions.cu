#include "kernels.cuh"
#include "functions.cuh"
#include "utils.hpp"

void conv2dCPU(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    int filterSize { 2 * radius + 1 };

    for (int row {0}; row < height; ++row) {
        for (int col {0}; col < width; ++col) {
            float pValue { 0.0f };

            // Clamp the filter window to valid input bounds (avoids inner if-branch)
            const int fRowStart { std::max(0, radius - row) };
            const int fRowEnd { std::min(filterSize, height + radius - row) };
            const int fColStart { std::max(0, radius - col) };
            const int fColEnd { std::min(filterSize, width  + radius - col) };

            for (int fRow { fRowStart }; fRow < fRowEnd; ++fRow) {
                const int inRow { row - radius + fRow };
                const int filterRowOff { fRow * filterSize };
                const int inRowOff { inRow * width };

                for (int fCol { fColStart }; fCol < fColEnd; ++fCol) {
                    const int inCol { col - radius + fCol };
                    pValue += filter[filterRowOff + fCol] * inArray[inRowOff + inCol];
                }
            }
            outArray[row * width + col] = pValue;
        }
    }

}

void conv2d(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    // Allocate device memory
    int filterDim { 2*radius + 1 };
    std::size_t arraySize { static_cast<std::size_t>(height * width * sizeof(float)) };
    std::size_t filterSize { static_cast<std::size_t>(filterDim * filterDim * sizeof(float)) };
    
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

void conv2dConstMem(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    // Initialize constant mememory
    int filterDim { 2*radius + 1 };
    std::size_t filterSize {static_cast<std::size_t>(filterDim * filterDim * sizeof(float))};
    CUDA_CHECK(uploadConstFilter(filter, filterSize));

    // Allocate device memory
    std::size_t arraySize { static_cast<std::size_t>(height * width * sizeof(float)) };

    float *inArray_d { nullptr }, *outArray_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inArray_d, arraySize));
    CUDA_CHECK(cudaMalloc(&outArray_d, arraySize));

    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(inArray_d, inArray, arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(outArray_d, outArray, arraySize, cudaMemcpyHostToDevice));

    // Call kernel
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 gridSize(
        static_cast<int>(utils::cdiv(width, BLOCK_SIZE)),
        static_cast<int>(utils::cdiv(height, BLOCK_SIZE)),
        1
    );
    conv2dKernelConstMem<<<gridSize, blockSize>>>(inArray_d, outArray_d, radius, height, width);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(outArray, outArray_d, arraySize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(inArray_d));
    CUDA_CHECK(cudaFree(outArray_d));
}

void conv2dTiledIn(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    // Initialize constant memory
    int filterDim { 2*radius + 1 };
    std::size_t filterSize {static_cast<std::size_t>(filterDim * filterDim * sizeof(float))};
    CUDA_CHECK(uploadConstFilter(filter, filterSize));

    // Allocate memory on device
    std::size_t arraySize { static_cast<std::size_t>(height * width * sizeof(float)) };

    float *inArray_d { nullptr }, *outArray_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inArray_d, arraySize));
    CUDA_CHECK(cudaMalloc(&outArray_d, arraySize));

    // Copy arrays to the device
    CUDA_CHECK(cudaMemcpy(inArray_d, inArray, arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(outArray_d, outArray, arraySize, cudaMemcpyHostToDevice));

    // Run the kernel
    const dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM);
    const dim3 gridSize(
        static_cast<int>(utils::cdiv(width + 2 * FILTER_RADIUS, OUT_TILE_DIM)),
        static_cast<int>(utils::cdiv(height + 2 * FILTER_RADIUS, OUT_TILE_DIM)),
        1
    );
    conv2dKernelTiledIn<<<gridSize, blockSize>>>(inArray_d, outArray_d, radius, height, width);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(outArray, outArray_d, arraySize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(inArray_d));
    CUDA_CHECK(cudaFree(outArray_d));
}

void conv2dTiledOut(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    // Initialize constant memory
    int filterDim { 2*radius + 1 };
    std::size_t filterSize {static_cast<std::size_t>(filterDim * filterDim * sizeof(float))};
    CUDA_CHECK(uploadConstFilter(filter, filterSize));

    // Allocate memory on device
    std::size_t arraySize { static_cast<std::size_t>(height * width * sizeof(float)) };

    float *inArray_d { nullptr }, *outArray_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inArray_d, arraySize));
    CUDA_CHECK(cudaMalloc(&outArray_d, arraySize));

    // Copy arrays to the device
    CUDA_CHECK(cudaMemcpy(inArray_d, inArray, arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(outArray_d, outArray, arraySize, cudaMemcpyHostToDevice));

    // Run the kernel
    const dim3 blockSize(OUT_TILE_DIM, OUT_TILE_DIM);
    const dim3 gridSize(
        static_cast<int>(utils::cdiv(width, OUT_TILE_DIM)),
        static_cast<int>(utils::cdiv(height, OUT_TILE_DIM)),
        1
    );
    conv2dKernelTiledOut<<<gridSize, blockSize>>>(inArray_d, outArray_d, radius, height, width);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(outArray, outArray_d, arraySize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(inArray_d));
    CUDA_CHECK(cudaFree(outArray_d));
}

void conv2dTiledCached(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    // Initialize constant memory
    int filterDim { 2*radius + 1 };
    std::size_t filterSize {static_cast<std::size_t>(filterDim * filterDim * sizeof(float))};
    CUDA_CHECK(uploadConstFilter(filter, filterSize));

    // Allocate memory on device
    std::size_t arraySize { static_cast<std::size_t>(height * width * sizeof(float)) };

    float *inArray_d { nullptr }, *outArray_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inArray_d, arraySize));
    CUDA_CHECK(cudaMalloc(&outArray_d, arraySize));

    // Copy arrays to the device
    CUDA_CHECK(cudaMemcpy(inArray_d, inArray, arraySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(outArray_d, outArray, arraySize, cudaMemcpyHostToDevice));

    // Run the kernel
    const dim3 blockSize(TILE_DIM, TILE_DIM);
    const dim3 gridSize(
        static_cast<int>(utils::cdiv(width, TILE_DIM)),
        static_cast<int>(utils::cdiv(height, TILE_DIM)),
        1
    );
    conv2dKernelTiledCached<<<gridSize, blockSize>>>(inArray_d, outArray_d, radius, height, width);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(outArray, outArray_d, arraySize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(inArray_d));
    CUDA_CHECK(cudaFree(outArray_d));
}
