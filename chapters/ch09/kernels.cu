#include "kernels.cuh"
#include <cuda_runtime.h>

__global__ void histogramKernel(
    const char* data,
    unsigned int length,
    unsigned int* histo
) {
    unsigned int i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i < length) {
        int alphabetPosition { data[i] - 'a' };
        if (alphabetPosition >= 0 && alphabetPosition < 26) {
            atomicAdd(&(histo[alphabetPosition/BIN_SIZE]), 1);
        }
    }
}

__global__ void histogramKernelPrivate(
    const char* data,
    unsigned int length,
    unsigned int* histo
) {
    unsigned int i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i < length) {
        int alphabetPosition { data[i] - 'a' };
        if (alphabetPosition >= 0 && alphabetPosition < 26) {
            atomicAdd(&(histo[blockIdx.x*NUM_BINS + alphabetPosition/BIN_SIZE]), 1);
        }
    }

    if (blockIdx.x > 0) {
        __syncthreads();

        for (unsigned int bin { threadIdx.x }; bin < NUM_BINS; bin += blockDim.x) {
            unsigned int binValue { histo[blockIdx.x*NUM_BINS + bin] };
            if (binValue > 0) {
                atomicAdd(&(histo[bin]), binValue);
            }
        }
    }
}

__global__ void histogramKernelSharedMem(
    const char* data,
    unsigned int length,
    unsigned int* histo
) {
    // Initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin { threadIdx.x }; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Histogram computation
    unsigned int i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i < length) {
        int alphabetPosition { data[i] - 'a' };
        if (alphabetPosition >= 0 && alphabetPosition < 26) {
            atomicAdd(&(histo_s[alphabetPosition/BIN_SIZE]), 1);
        }
    }
    __syncthreads();

    // Commit ot global memory
    for (unsigned int bin { threadIdx.x }; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue { histo_s[bin] };
        if (binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}
