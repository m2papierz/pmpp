#include "utils.hpp"
#include "stencil_sweep.cuh"
#include <cuda_runtime.h>

void stencilCPU(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    for (int dep { 1 }; dep < n - 1; ++dep) {
        for (int row { 1 }; row < n - 1; ++row) {
            for (int col { 1 }; col < n - 1; ++col) {
                out[dep*n*n + row*n + col] = c0*in[dep*n*n + row*n + col]+
                                            c1*in[dep*n*n + row*n + (col - 1)] +
                                            c2*in[dep*n*n + row*n + (col + 1)] +
                                            c3*in[dep*n*n + (row - 1)*n + col] +
                                            c4*in[dep*n*n + (row + 1)*n + col] +
                                            c5*in[(dep - 1)*n*n + row*n + col] +
                                            c6*in[(dep + 1)*n*n + row*n + col];
            }
        }
    }
}

__global__ void stencilKernelNaive(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    int dep { static_cast<int>(blockIdx.z*blockDim.z + threadIdx.z) };
    int row { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int col { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };

    if (
        dep >= 1 && dep < n - 1 &&
        row >= 1 && row < n - 1 &&
        col >= 1 && col < n - 1
    ) {
        out[dep*n*n + row*n + col] = c0*in[dep*n*n + row*n + col]+
                                     c1*in[dep*n*n + row*n + (col - 1)] +
                                     c2*in[dep*n*n + row*n + (col + 1)] +
                                     c3*in[dep*n*n + (row - 1)*n + col] +
                                     c4*in[dep*n*n + (row + 1)*n + col] +
                                     c5*in[(dep - 1)*n*n + row*n + col] +
                                     c6*in[(dep + 1)*n*n + row*n + col];
    }
}

void stencilNaive(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    std::size_t tensorSize { static_cast<std::size_t>(n*n*n*sizeof(float)) };

    // Allocate device memory
    float *in_d { nullptr }, *out_d { nullptr };
    CUDA_CHECK(cudaMalloc(&in_d, tensorSize));
    CUDA_CHECK(cudaMalloc(&out_d, tensorSize));

    // Move data from host to the device
    CUDA_CHECK(cudaMemcpy(in_d, in, tensorSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_d, out, tensorSize, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 blockSize(OUT_TILE_DIM, OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 gridSize(
        utils::cdiv(n, OUT_TILE_DIM),
        utils::cdiv(n, OUT_TILE_DIM),
        utils::cdiv(n, OUT_TILE_DIM)
    );
    stencilKernelNaive<<<gridSize, blockSize>>>(in_d, out_d, n, c0, c1, c2, c3, c4, c5, c6);

    // Check cuda errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(out, out_d, tensorSize, cudaMemcpyDeviceToHost));

    // Free devuce memory
    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));
}

__global__ void stencilKernelSharedMem(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    int dep { static_cast<int>(blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1) };
    int row { static_cast<int>(blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1) };
    int col { static_cast<int>(blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1) };

    // Collaboratively load data into shared memory
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (dep >= 0 && dep < n && row >= 0 && row < n && col >= 0 && col < n) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[dep*n*n + row*n + col];
    }
    __syncthreads();

    if (dep >= 1 && dep < n - 1 && row >= 1 && row < n - 1 && col >= 1 && col < n - 1) {
        if (
            threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1
        ) {
            out[dep*n*n + row*n + col] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]+
                                         c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                                         c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                         c3*in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                                         c4*in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                         c5*in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                                         c6*in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

void stencilSharedMem(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    std::size_t tensorSize { static_cast<std::size_t>(n*n*n*sizeof(float)) };

    // Allocate device memory
    float *in_d { nullptr }, *out_d { nullptr };
    CUDA_CHECK(cudaMalloc(&in_d, tensorSize));
    CUDA_CHECK(cudaMalloc(&out_d, tensorSize));

    // Move data from host to the device
    CUDA_CHECK(cudaMemcpy(in_d, in, tensorSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_d, out, tensorSize, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 blockSize(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridSize(
        utils::cdiv(n, OUT_TILE_DIM),
        utils::cdiv(n, OUT_TILE_DIM),
        utils::cdiv(n, OUT_TILE_DIM)
    );
    stencilKernelSharedMem<<<gridSize, blockSize>>>(in_d, out_d, n, c0, c1, c2, c3, c4, c5, c6);

    // Check cuda errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(out, out_d, tensorSize, cudaMemcpyDeviceToHost));

    // Free devuce memory
    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));
}

__global__ void stencilKernelThreadCoarsening(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    int depStart { static_cast<int>(blockIdx.z*OUT_TILE_DIM_TC) };
    int row { static_cast<int>(blockIdx.y*OUT_TILE_DIM_TC + threadIdx.y - 1) };
    int col { static_cast<int>(blockIdx.x*OUT_TILE_DIM_TC + threadIdx.x - 1) };

    // Collaboratively load data into shared memory
    __shared__ float inPrev_s[IN_TILE_DIM_TC][IN_TILE_DIM_TC];
    __shared__ float inCurr_s[IN_TILE_DIM_TC][IN_TILE_DIM_TC];
    __shared__ float inNext_s[IN_TILE_DIM_TC][IN_TILE_DIM_TC];

    if (depStart - 1 >= 0 && depStart - 1 < n && row >= 0 && row < n && col >= 0 && col < n) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(depStart - 1)*n*n + row*n + col];
    }
    if (depStart >= 0 && depStart < n && row >= 0 && row < n && col >= 0 && col < n) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[depStart*n*n + row*n + col];
    }

    for (int dep { depStart }; dep < depStart + OUT_TILE_DIM_TC; ++dep) {
        if(dep + 1 >= 0 && dep + 1 < n && row >= 0 && row < n && col >= 0 && col < n) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(dep + 1)*n*n + row*n + col];
        }
        __syncthreads();

        if (dep >= 1 && dep < n - 1 && row >= 1 && row < n - 1 && col >= 1 && col < n - 1) {
            if(
                threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_TC - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_TC - 1
            ) {
                out[dep*n*n + row*n + col] = c0*inCurr_s[threadIdx.y][threadIdx.x] +
                                             c1*inCurr_s[threadIdx.y][threadIdx.x - 1] +
                                             c2*inCurr_s[threadIdx.y][threadIdx.x + 1] +
                                             c3*inCurr_s[threadIdx.y - 1][threadIdx.x] +
                                             c4*inCurr_s[threadIdx.y + 1][threadIdx.x] +
                                             c5*inPrev_s[threadIdx.y][threadIdx.x] +
                                             c6*inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
    
}

void stencilThreadCoarsening(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    std::size_t tensorSize { static_cast<std::size_t>(n*n*n*sizeof(float)) };

    // Allocate device memory
    float *in_d { nullptr }, *out_d { nullptr };
    CUDA_CHECK(cudaMalloc(&in_d, tensorSize));
    CUDA_CHECK(cudaMalloc(&out_d, tensorSize));

    // Move data from host to the device
    CUDA_CHECK(cudaMemcpy(in_d, in, tensorSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_d, out, tensorSize, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 blockSize(IN_TILE_DIM_TC, IN_TILE_DIM_TC, 1);
    dim3 gridSize(
        utils::cdiv(n, OUT_TILE_DIM_TC),
        utils::cdiv(n, OUT_TILE_DIM_TC),
        utils::cdiv(n, OUT_TILE_DIM_TC)
    );
    stencilKernelThreadCoarsening<<<gridSize, blockSize>>>(in_d, out_d, n, c0, c1, c2, c3, c4, c5, c6);

    // Check cuda errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(out, out_d, tensorSize, cudaMemcpyDeviceToHost));

    // Free devuce memory
    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));
}

__global__ void stencilKernelRegisterTiling(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    int depStart { static_cast<int>(blockIdx.z*OUT_TILE_DIM_RT) };
    int row { static_cast<int>(blockIdx.y*OUT_TILE_DIM_RT + threadIdx.y - 1) };
    int col { static_cast<int>(blockIdx.x*OUT_TILE_DIM_RT + threadIdx.x - 1) };
    float inPrev {};

    // Collaboratively load data into shared memory
    __shared__ float inCurr_s[IN_TILE_DIM_RT][IN_TILE_DIM_RT];
    float inCurr {};
    float inNext {};

    if (depStart - 1 >= 0 && depStart - 1 < n && row >= 0 && row < n && col >= 0 && col < n) {
        inPrev = in[(depStart - 1)*n*n + row*n + col];
    }
    if (depStart >= 0 && depStart < n && row >= 0 && row < n && col >= 0 && col < n) {
        inCurr = in[depStart*n*n + row*n + col];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for (int dep { depStart }; dep < depStart + OUT_TILE_DIM_RT; ++dep) {
        if(dep + 1 >= 0 && dep + 1 < n && row >= 0 && row < n && col >= 0 && col < n) {
            inNext = in[(dep + 1)*n*n + row*n + col];
        }
        __syncthreads();

        if (dep >= 1 && dep < n - 1 && row >= 1 && row < n - 1 && col >= 1 && col < n - 1) {
            if(
                threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_RT - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_RT - 1
            ) {
                out[dep*n*n + row*n + col] = c0*inCurr +
                                             c1*inCurr_s[threadIdx.y][threadIdx.x - 1] +
                                             c2*inCurr_s[threadIdx.y][threadIdx.x + 1] +
                                             c3*inCurr_s[threadIdx.y - 1][threadIdx.x] +
                                             c4*inCurr_s[threadIdx.y + 1][threadIdx.x] +
                                             c5*inPrev +
                                             c6*inNext;
            }
        }
        __syncthreads();

        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

void stencilRegisterTiling(
    float* in, float* out,
    unsigned int n,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6
) {
    std::size_t tensorSize { static_cast<std::size_t>(n*n*n*sizeof(float)) };

    // Allocate device memory
    float *in_d { nullptr }, *out_d { nullptr };
    CUDA_CHECK(cudaMalloc(&in_d, tensorSize));
    CUDA_CHECK(cudaMalloc(&out_d, tensorSize));

    // Move data from host to the device
    CUDA_CHECK(cudaMemcpy(in_d, in, tensorSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_d, out, tensorSize, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 blockSize(IN_TILE_DIM_RT, IN_TILE_DIM_RT, 1);
    dim3 gridSize(
        utils::cdiv(n, OUT_TILE_DIM_RT),
        utils::cdiv(n, OUT_TILE_DIM_RT),
        utils::cdiv(n, OUT_TILE_DIM_RT)
    );
    stencilKernelRegisterTiling<<<gridSize, blockSize>>>(in_d, out_d, n, c0, c1, c2, c3, c4, c5, c6);

    // Check cuda errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(out, out_d, tensorSize, cudaMemcpyDeviceToHost));

    // Free device  memory
    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));
}
