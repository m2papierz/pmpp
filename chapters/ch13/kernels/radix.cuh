#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define NUM_BITS 32
#define BLOCK_SIZE 1024
#define RADIX_BITS 4
#define COARSE_FACTOR 2

using BlockScanT = cub::BlockScan<unsigned int, BLOCK_SIZE>;

__global__ void computeBitsKernel(
    const unsigned int* inputArr,
    unsigned int* bits,
    const unsigned int n,
    const unsigned int iter
);

__global__ void scatterKernel(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int* bits,
    const unsigned int n,
    const unsigned int iter
);

__global__ void localSortKernel(
    const unsigned int* inputArr,
    unsigned int* localScan,
    unsigned int* blockZeros,
    unsigned int* blockOnes,
    const unsigned int n,
    unsigned int iter
);

__global__ void scatterCoalescedKernel(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int* localScan,
    const unsigned int* blockZeroOffsets,
    const unsigned int* blockOneOffsets,
    const unsigned int totalZeros,
    const unsigned int n,
    const unsigned int iter
);

__global__ void localSortMultibitKernel(
    const unsigned int* inputArr,
    unsigned int* localScan,
    unsigned int* blockBucketCounts,
    unsigned int* bucketTotals,
    const unsigned int n,
    const unsigned int shift,
    const unsigned int numBlocks,
    const unsigned int radix
);

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
);

__global__ void localSortCoarseKernel(
    const unsigned int* inputArr,
    unsigned int* localScan,
    unsigned int* blockZeros,
    unsigned int* blockOnes,
    const unsigned int n,
    const unsigned int iter
);

__global__ void scatterCoalescedCoarseKernel(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int* localScan,
    const unsigned int* blockZeroOffsets,
    const unsigned int* blockOneOffsets,
    const unsigned int totalZeros,
    const unsigned int n,
    const unsigned int iter
);
