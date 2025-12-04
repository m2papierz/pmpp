#pragma once

#define NUM_BITS 32
#define BLOCK_SIZE 1024

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
