#include "radix.cuh"

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

__global__ void localSortKernel(
    const unsigned int* inputArr,
    unsigned int* localScan,
    unsigned int* blockZeros,
    unsigned int* blockOnes,
    const unsigned int n,
    unsigned int iter
) {
    unsigned int tid { threadIdx.x };
    unsigned int blockBase { blockIdx.x*blockDim.x };
    unsigned int gid {blockBase + tid };

    if (blockBase >= n) return;

    // How many elements actually belong to this block
    const unsigned int active { min(blockDim.x, n - blockBase) };

    // CUB shared temp storage
    __shared__ typename BlockScanT::TempStorage scanTemp;

    // Per-thread data
    unsigned int bit { 0 };
    if (gid  < n) {
        bit = (inputArr[gid] >> iter) & 1u;
    }

    // Each thread contributes its bit (0 or 1) to the scan
    unsigned int numOnesBefore { 0 };
    unsigned int numOnesTotal { 0 };
    unsigned int in { (tid < active) ? bit : 0 };

    BlockScanT(scanTemp).ExclusiveSum(in, numOnesBefore, numOnesTotal);

    if (gid < n) {
        localScan[gid] = numOnesBefore;
    }

    if (tid == 0) {
        blockZeros[blockIdx.x] = active - numOnesTotal;
        blockOnes[blockIdx.x]  = numOnesTotal;
    }
}

__global__ void scatterCoalescedKernel(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int* localScan,
    const unsigned int* blockZeroOffsets,
    const unsigned int* blockOneOffsets,
    const unsigned int totalZeros,
    const unsigned int n,
    const unsigned int iter
) {
    unsigned int tid { threadIdx.x };
    unsigned int gid { blockIdx.x*blockDim.x + tid };

    if (gid < n) {
        unsigned int key { inputArr[gid] };
        unsigned int bit { (key >> iter) & 1 };
        unsigned int localPrefix { localScan[gid] };
    
        unsigned int dest{};
        if (bit == 0) {
            dest = blockZeroOffsets[blockIdx.x] + tid - localPrefix;
        } else {
            dest = totalZeros + blockOneOffsets[blockIdx.x] + localPrefix;
        }

        outputArr[dest] = key;
    }
}
