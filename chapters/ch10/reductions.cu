#include "utils.hpp"
#include "kernels.cuh"
#include "reductions.cuh"
#include <cmath>
#include <limits>

void reductionSumSerial(const float* inputData, float* outputData, int n) {
    float sum { 0.0f };
    for (int i { 0 }; i < n; ++i) {
        sum += inputData[i];
    }
    *outputData = sum;
}

void reductionMaxSerial(const float* inputData, float* outputData, int n) {
    float maxVal { std::numeric_limits<float>::lowest() };
    for (int i { 0 }; i < n; ++i) {
        maxVal = std::fmax(maxVal, inputData[i]);
    }
    *outputData = maxVal;
}

void reductionSimple(float* inputData, float* outputData, int n) {
    std::size_t inputSize { static_cast<std::size_t>(n*sizeof(float)) };
    std::size_t outputSize { static_cast<std::size_t>(sizeof(float)) };

    float *inputData_d { nullptr }, *outputData_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputData_d, inputSize));
    CUDA_CHECK(cudaMalloc(&outputData_d, outputSize));

    CUDA_CHECK(cudaMemcpy(inputData_d, inputData, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputData_d, 0, outputSize));

    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(1, 1, 1);
    simpleKernel<<<gridDim, blockDim>>>(inputData_d, outputData_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputData, outputData_d, outputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputData_d));
    CUDA_CHECK(cudaFree(outputData_d));
}

void reductionConvergent(float* inputData, float* outputData, int n) {
    std::size_t inputSize { static_cast<std::size_t>(n*sizeof(float)) };
    std::size_t outputSize { static_cast<std::size_t>(sizeof(float)) };

    float *inputData_d { nullptr }, *outputData_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputData_d, inputSize));
    CUDA_CHECK(cudaMalloc(&outputData_d, outputSize));

    CUDA_CHECK(cudaMemcpy(inputData_d, inputData, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputData_d, 0, outputSize));

    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(1, 1, 1);
    convergentKernel<<<gridDim, blockDim>>>(inputData_d, outputData_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputData, outputData_d, outputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputData_d));
    CUDA_CHECK(cudaFree(outputData_d));
}

void reductionConvergentSharedMem(float* inputData, float* outputData, int n) {
    std::size_t inputSize { static_cast<std::size_t>(n*sizeof(float)) };
    std::size_t outputSize { static_cast<std::size_t>(sizeof(float)) };

    float *inputData_d { nullptr }, *outputData_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputData_d, inputSize));
    CUDA_CHECK(cudaMalloc(&outputData_d, outputSize));

    CUDA_CHECK(cudaMemcpy(inputData_d, inputData, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputData_d, 0, outputSize));

    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(1, 1, 1);
    convergentSharedMemKernel<<<gridDim, blockDim>>>(inputData_d, outputData_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputData, outputData_d, outputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputData_d));
    CUDA_CHECK(cudaFree(outputData_d));
}

void reductionSegmented(const float* inputData, float* outputData, int n) {
    std::size_t inputSize { static_cast<std::size_t>(n*sizeof(float)) };
    std::size_t outputSize { static_cast<std::size_t>(sizeof(float)) };

    float *inputData_d { nullptr }, *outputData_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputData_d, inputSize));
    CUDA_CHECK(cudaMalloc(&outputData_d, outputSize));

    CUDA_CHECK(cudaMemcpy(inputData_d, inputData, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputData_d, 0, outputSize));

    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(utils::cdiv(n, 2*BLOCK_DIM), 1, 1);
    segmentedKernel<<<gridDim, blockDim>>>(inputData_d, outputData_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputData, outputData_d, outputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputData_d));
    CUDA_CHECK(cudaFree(outputData_d));
}

void reductionCoarsed(const float* inputData, float* outputData, int n) {
    std::size_t inputSize { static_cast<std::size_t>(n*sizeof(float)) };
    std::size_t outputSize { static_cast<std::size_t>(sizeof(float)) };

    float *inputData_d { nullptr }, *outputData_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputData_d, inputSize));
    CUDA_CHECK(cudaMalloc(&outputData_d, outputSize));

    CUDA_CHECK(cudaMemcpy(inputData_d, inputData, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputData_d, 0, outputSize));

    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(utils::cdiv(n, 2*COARSE_FACTOR*BLOCK_DIM), 1, 1);
    coarsenedKernel<<<gridDim, blockDim>>>(inputData_d, outputData_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputData, outputData_d, outputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputData_d));
    CUDA_CHECK(cudaFree(outputData_d));
}

void reductionMaxCoarsed(const float* inputData, float* outputData, int n) {
    std::size_t inputSize { static_cast<std::size_t>(n*sizeof(float)) };
    std::size_t outputSize { static_cast<std::size_t>(sizeof(float)) };

    float *inputData_d { nullptr }, *outputData_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputData_d, inputSize));
    CUDA_CHECK(cudaMalloc(&outputData_d, outputSize));

    CUDA_CHECK(cudaMemcpy(inputData_d, inputData, inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(outputData_d, 0, outputSize));

    dim3 blockDim(BLOCK_DIM, 1, 1);
    dim3 gridDim(utils::cdiv(n, 2*COARSE_FACTOR*BLOCK_DIM), 1, 1);
    coarsenedKernelMax<<<gridDim, blockDim>>>(inputData_d, outputData_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(outputData, outputData_d, outputSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputData_d));
    CUDA_CHECK(cudaFree(outputData_d));
}
