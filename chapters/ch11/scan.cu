#include "kernels.cuh"
#include "scan.cuh"
#include "utils.hpp"
#include <assert.h>

void scanSequential(const float* x, float* y, unsigned int n) {
    y[0] = x[0];
    for(int i { 1 }; i < n; ++i) {
        y[i] = y[i - 1] + x[i];
    }
}

void koggeStone(const float* x, float* y, unsigned int n) {
    std::size_t seqSize { static_cast<std::size_t>(n*sizeof(float)) };

    float *x_d { nullptr }, *y_d { nullptr };
    CUDA_CHECK(cudaMalloc(&x_d, seqSize));
    CUDA_CHECK(cudaMalloc(&y_d, seqSize));

    CUDA_CHECK(cudaMemcpy(x_d, x, seqSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(y_d, 0, seqSize));

    dim3 blockSize(n);
    dim3 gridSize(1);
    koggeStoneKernel<<<gridSize, blockSize>>>(x_d, y_d, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y, y_d, seqSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
}

void koggeStoneDoubleBuffer(const float* x, float* y, unsigned int n) {
    std::size_t seqSize { static_cast<std::size_t>(n*sizeof(float)) };

    float *x_d { nullptr }, *y_d { nullptr };
    CUDA_CHECK(cudaMalloc(&x_d, seqSize));
    CUDA_CHECK(cudaMalloc(&y_d, seqSize));

    CUDA_CHECK(cudaMemcpy(x_d, x, seqSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(y_d, 0, seqSize));

    dim3 blockSize(n);
    dim3 gridSize(1);
    koggeStoneKernelDoubleBuffer<<<gridSize, blockSize>>>(x_d, y_d, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y, y_d, seqSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
}

void brentKung(const float* x, float* y, unsigned int n) {
    std::size_t seqSize { static_cast<std::size_t>(n*sizeof(float)) };

    float *x_d { nullptr }, *y_d { nullptr };
    CUDA_CHECK(cudaMalloc(&x_d, seqSize));
    CUDA_CHECK(cudaMalloc(&y_d, seqSize));

    CUDA_CHECK(cudaMemcpy(x_d, x, seqSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(y_d, 0, seqSize));

    dim3 blockSize(n);
    dim3 gridSize(1);
    brentKungKernel<<<gridSize, blockSize>>>(x_d, y_d, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y, y_d, seqSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
}

void coarsenedThreePhase(const float* x, float* y, unsigned int n) {
    std::size_t seqSize { static_cast<std::size_t>(n*sizeof(float)) };

    float *x_d { nullptr }, *y_d { nullptr };
    CUDA_CHECK(cudaMalloc(&x_d, seqSize));
    CUDA_CHECK(cudaMalloc(&y_d, seqSize));

    CUDA_CHECK(cudaMemcpy(x_d, x, seqSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(y_d, 0, seqSize));

    dim3 blockSize(utils::cdiv(n, COARSE_FACTOR));
    dim3 gridSize(1);
    std::size_t sharedMemSize { (n + n / COARSE_FACTOR) * sizeof(float) };
    coarsenedThreePhaseKernel<<<gridSize, blockSize, sharedMemSize>>>(x_d, y_d, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y, y_d, seqSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
}
