#include "radix.cuh"
#include <cuda_runtime.h>

__global__ void computeBitsKernel(
    const unsigned int* inputArr,
    unsigned int* bits,
    const unsigned int n,
    const unsigned int iter
) {
    unsigned int i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i < n) {
        unsigned int key = inputArr[i];
        bits[i] = (key >> iter) & 1u;
    }
}

__global__ void scatterKernel(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int* bits,
    const unsigned int n,
    const unsigned int iter
) {
    unsigned int i { blockIdx.x * blockDim.x + threadIdx.x };
    if (i >= n) return;

    unsigned int key { inputArr[i] };
    unsigned int bit { (key >> iter) & 1u };

    unsigned int numOnesBefore { bits[i] };
    unsigned int numOnesTotal { bits[n] }; 
    unsigned int dest = {
        (bit == 0)
            ? (i - numOnesBefore)
            : (n - numOnesTotal + numOnesBefore)
    };

    outputArr[dest] = key;
}
