#include "kernels.cuh"

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
    inputData_s[i] = inputData[i] + inputData[i + 1];
    
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

__global__ void segmentedKernel(float* inputData, float* outputData) {
    unsigned int segment { 2*blockDim.x*blockIdx.x };
    unsigned int i { segment + threadIdx.x };
    unsigned int t { threadIdx.x };

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

__global__ void coarsenedKernel(float* inputData, float* outputData) {
    unsigned int segment { 2*blockDim.x*blockIdx.x*COARSE_FACTOR };
    unsigned int i { segment + threadIdx.x };
    unsigned int t { threadIdx.x };


    __shared__ float inputData_s[BLOCK_DIM];
    float sum { inputData[i] };
    for (unsigned int tile {1}; tile < COARSE_FACTOR*2; ++tile) {
        sum += inputData[i + tile*BLOCK_DIM];
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
