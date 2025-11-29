#include "kernels.cuh"
#include "scan.cuh"
#include "utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <assert.h>

void scanSequential(const float* x, float* y, unsigned int n) {
    y[0] = x[0];
    for(int i { 1 }; i < n; ++i) {
        y[i] = y[i - 1] + x[i];
    }
}

void thrustScan(const float* x, float* y, unsigned int n) {
    // Allocate device memory
    thrust::device_vector<float> d_in(n);
    thrust::device_vector<float> d_out(n);

    // Copy input from host -> device
    thrust::copy(x, x + n, d_in.begin());

    // Inclusive scan on device
    thrust::inclusive_scan(
        thrust::device,
        d_in.begin(),
        d_in.end(),
        d_out.begin()
    );

    // Copy result back to host
    thrust::copy(d_out.begin(), d_out.end(), y);
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

void hierarchicalScan(const float* x, float* y, unsigned int n) {
    dim3 blockSize(SECTION_SIZE);
    dim3 gridSizePhase1(utils::cdiv(n, SECTION_SIZE));
    unsigned int numBlocks { gridSizePhase1.x };
    dim3 gridSizePhase2(utils::cdiv(numBlocks, SECTION_SIZE));

    std::size_t seqSize { static_cast<std::size_t>(n*sizeof(float)) };
    std::size_t sharedMemSize { SECTION_SIZE * sizeof(float) };

    float *x_d { nullptr }, *y_d { nullptr }, *s_d { nullptr };
    CUDA_CHECK(cudaMalloc(&x_d, seqSize));
    CUDA_CHECK(cudaMalloc(&y_d, seqSize));
    CUDA_CHECK(cudaMalloc(&s_d, numBlocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(x_d, x, seqSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(y_d, 0, seqSize));
    CUDA_CHECK(cudaMemset(s_d, 0, numBlocks * sizeof(float)));

    // Phase 1
    hierarchicalKernelPhase1<<<gridSizePhase1, blockSize, sharedMemSize>>>(x_d, y_d, s_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //Phase 2
    hierarchicalKernelPhase2<<<gridSizePhase2, blockSize, sharedMemSize>>>(s_d, numBlocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phase 3
    hierarchicalKernelPhase3<<<gridSizePhase1, blockSize>>>(y_d, s_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y, y_d, seqSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(s_d));
}

void hierarchicalDominoScan(const float* x, float* y, unsigned int n) {
    dim3 blockSize(SECTION_SIZE);
    dim3 gridSize(utils::cdiv(n, SECTION_SIZE));
    std::size_t sharedMemSize { SECTION_SIZE * sizeof(float) };
    std::size_t seqSize { static_cast<std::size_t>(n*sizeof(float)) };
    unsigned int numBlocks { gridSize.x };

    int *flags_d { nullptr }, *block_counter_d { nullptr };
    float *x_d { nullptr }, *y_d { nullptr }, *scan_vals_d { nullptr };
    CUDA_CHECK(cudaMalloc(&x_d, seqSize));
    CUDA_CHECK(cudaMalloc(&y_d, seqSize));
    CUDA_CHECK(cudaMalloc(&scan_vals_d, (numBlocks + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flags_d, (numBlocks + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&block_counter_d, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(x_d, x, seqSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(flags_d, 0, (numBlocks + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(block_counter_d, 0, sizeof(int)));

    
    hierarchicalDominoKernel<<<gridSize, blockSize, sharedMemSize>>>(x_d, y_d, scan_vals_d, flags_d, block_counter_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(y, y_d, seqSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(scan_vals_d));
    CUDA_CHECK(cudaFree(flags_d));
    CUDA_CHECK(cudaFree(block_counter_d));
}
