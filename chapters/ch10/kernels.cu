#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>

__global__ void simpleKernel(float* inputData, float* outputData) {
    unsigned int i { 2*threadIdx.x };
    for (unsigned int stride { 1 }; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            inputData[i] += inputData[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *outputData = inputData[0];
    }
}

__global__ void convergentKernel(float* inputData, float* outputData) {
    unsigned int i { threadIdx.x };
    for (unsigned int stride { blockDim.x }; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            inputData[i] += inputData[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *outputData = inputData[0];
    }
}

__global__ void convergentSharedMemKernel(float* inputData, float* outputData) {
    unsigned int i { threadIdx.x };

    __shared__ float inputData_s[BLOCK_DIM];
    inputData_s[i] = inputData[i] + inputData[i + BLOCK_DIM];
    
    for (unsigned int stride { blockDim.x/2 }; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            inputData_s[i] += inputData_s[i + stride];
        }
    }

    if (threadIdx.x == 0) {
        *outputData = inputData_s[0];
    }
}

__global__ void segmentedKernel(const float* inputData, float* outputData, int length) {
    unsigned int segment { 2*blockDim.x*blockIdx.x };
    unsigned int i { segment + threadIdx.x };
    unsigned int t { threadIdx.x };

    if (i >= length || i + BLOCK_DIM >= length) {
        return;
    }

    __shared__ float inputData_s[BLOCK_DIM];
    inputData_s[t] = inputData[i] + inputData[i + BLOCK_DIM];
    
    for (unsigned int stride { blockDim.x/2 }; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            inputData_s[t] += inputData_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(outputData, inputData_s[0]);
    }
}

__global__ void coarsenedKernel(const float* inputData, float* outputData, int length) {
    unsigned int segment { 2*blockDim.x*blockIdx.x*COARSE_FACTOR };
    unsigned int i { segment + threadIdx.x };
    unsigned int t { threadIdx.x };

    __shared__ float inputData_s[BLOCK_DIM];

    float sum { 0.0f };
    if (i < length) {
        sum = inputData[i];
        for (unsigned int tile { 1 }; tile < 2*COARSE_FACTOR; ++tile) {
            if (i + tile * BLOCK_DIM < length) {
                sum += inputData[i + tile*BLOCK_DIM];
            }
        }
    }
    inputData_s[t] = sum;

    for (unsigned int stride { blockDim.x/2 }; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            inputData_s[t] += inputData_s[t + stride];
        }
    }

    if (t == 0) {
        atomicAdd(outputData, inputData_s[0]);
    }
}

__global__ void coarsenedKernelMax(const float* inputData, float* outputData, int length) {
    unsigned int segment { 2*blockDim.x*blockIdx.x*COARSE_FACTOR };
    unsigned int i { segment + threadIdx.x };
    unsigned int t { threadIdx.x };

    __shared__ float inputData_s[BLOCK_DIM];

    float maxVal {inputData[i] };
    if (i < length) {
        for (unsigned int tile { 1 }; tile < 2*COARSE_FACTOR; ++tile) {
            if (i + tile * BLOCK_DIM < length) {
                maxVal = std::fmax(maxVal, inputData[i + tile*BLOCK_DIM]);
            }
        }
    }
    inputData_s[t] = maxVal;

    for (unsigned int stride { blockDim.x/2 }; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            inputData_s[t] = std::fmax(inputData_s[t], inputData_s[t + stride]);
        }
    }

    if (t == 0) {
        atomicExch(outputData, std::fmax(*outputData, inputData_s[0]));
    }
}

