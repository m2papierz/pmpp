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

__global__ void localSortMultibitKernel(
    const unsigned int* inputArr,
    unsigned int* localScan,
    unsigned int* blockBucketCounts,
    unsigned int* bucketTotals,
    const unsigned int n,
    const unsigned int shift,
    const unsigned int numBlocks,
    const unsigned int radix
) {
    unsigned int tid { threadIdx.x };
    unsigned int block { blockIdx.x };
    unsigned int blockBase { block*blockDim.x };
    unsigned int gid {blockBase + tid };

    if (blockBase >= n) return;

    bool isActive = (gid < n);

    unsigned int key { 0 };
    unsigned int bucket { 0 };
    if (isActive) {
        key = inputArr[gid];
        bucket = (key >> shift) & (radix - 1u);
    }

    // Per-block histogram in shared memory
    __shared__ typename BlockScanT::TempStorage scanTemp;

    // Initialize shared histogram
    for (unsigned int b { 0 }; b < radix; ++b) {
        unsigned int flag = (isActive && (bucket == b)) ? 1u : 0u;

        unsigned int prefix { 0 };
        unsigned int total  { 0 };
        BlockScanT(scanTemp).ExclusiveSum(flag, prefix, total);

        // Threads belonging to bucket b store their local index
        if (isActive && (bucket == b)) {
            localScan[gid] = prefix;
        }

        // One thread per block writes per-block count and updates global totals
        if (tid == 0) {
            blockBucketCounts[b * numBlocks + block] = total;
            if (total > 0) {
                atomicAdd(&bucketTotals[b], total);
            }
        }
        __syncthreads();
    }
}

__global__ void scatterCoalescedMultibitKernel(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int* localScan,
    const unsigned int* blockBucketOffsets,
    const unsigned int* bucketStarts,
    const unsigned int n,
    const unsigned int shift,
    const unsigned int numBlocks,
    const unsigned int radix
) {
    unsigned int tid { threadIdx.x };
    unsigned int block { blockIdx.x };
    unsigned int gid { block*blockDim.x + tid };

    if (gid >= n) return;

    unsigned int key { inputArr[gid] };
    unsigned int bucket { (key >> shift) & (radix - 1u) };

    unsigned int localIdx { localScan[gid] };
    unsigned int blockOffset { blockBucketOffsets[bucket*numBlocks + block] };
    unsigned int globalBucketStart { bucketStarts[bucket] };

    unsigned int dest { globalBucketStart + blockOffset + localIdx };
    outputArr[dest] = key;
}

__global__ void localSortCoarseKernel(
    const unsigned int* inputArr,
    unsigned int* localScan,
    unsigned int* blockZeros,
    unsigned int* blockOnes,
    const unsigned int n,
    const unsigned int iter
) {
    unsigned int tid { threadIdx.x };
    unsigned int block { blockIdx.x };
    unsigned int threads { blockDim.x };
    unsigned int blockBase{ block * threads * COARSE_FACTOR };

    if (blockBase >= n) return;

    unsigned int maxBlockElems { threads * COARSE_FACTOR };
    unsigned int blockElems{ min(maxBlockElems, n - blockBase) };

    __shared__ typename BlockScanT::TempStorage scanTemp;

    // First pass: compute bits for this thread's elements and total ones in this thread
    unsigned int onesInThread = 0;
    unsigned int bits[COARSE_FACTOR];

    for (unsigned int k = 0; k < COARSE_FACTOR; ++k) {
        unsigned int logicalIdx = tid * COARSE_FACTOR + k;
        unsigned int gid = blockBase + logicalIdx;

        if (logicalIdx < blockElems) {
            unsigned int key = inputArr[gid];
            unsigned int bit = (key >> iter) & 1u;
            bits[k] = bit;
            onesInThread += bit;
        } else {
            bits[k] = 0;
        }
    }

    // Block-wide exclusive scan on per-thread ones
    unsigned int threadOnesBefore { 0 };
    unsigned int totalOnesInBlock { 0 };
    BlockScanT(scanTemp).ExclusiveSum(onesInThread, threadOnesBefore, totalOnesInBlock);

    // Second pass: assign per-element localScan = #ones before this element in block
    unsigned int runningOnes = threadOnesBefore;

    for (unsigned int k = 0; k < COARSE_FACTOR; ++k) {
        unsigned int logicalIdx = tid * COARSE_FACTOR + k;
        if (logicalIdx >= blockElems) break;

        unsigned int gid = blockBase + logicalIdx;

        // onesBefore for this element
        localScan[gid] = runningOnes;

        // update running ones count
        if (bits[k] == 1u) {
            ++runningOnes;
        }
    }

    // One thread writes block counts
    if (tid == 0) {
        blockOnes[block] = totalOnesInBlock;
        blockZeros[block] = blockElems - totalOnesInBlock;
    }
}

__global__ void scatterCoalescedCoarseKernel(
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
    unsigned int block { blockIdx.x };
    unsigned int threads { blockDim.x };
    unsigned int blockBase{ block * threads * COARSE_FACTOR };

    if (blockBase >= n) return;

    const unsigned int maxBlockElems { threads * COARSE_FACTOR };
    const unsigned int blockElems { min(maxBlockElems, n - blockBase) };

    for (unsigned int k = 0; k < COARSE_FACTOR; ++k) {
        unsigned int logicalIdx = tid * COARSE_FACTOR + k;
        if (logicalIdx >= blockElems) break;

        unsigned int gid = blockBase + logicalIdx;
        unsigned int key = inputArr[gid];
        unsigned int bit = (key >> iter) & 1u;
        unsigned int onesBefore = localScan[gid];

        unsigned int dest;
        if (bit == 0u) {
            // zerosBefore = logicalIdx - onesBefore
            dest = blockZeroOffsets[block] + (logicalIdx - onesBefore);
        } else {
            dest = totalZeros + blockOneOffsets[block] + onesBefore;
        }
        outputArr[dest] = key;
    }
}
