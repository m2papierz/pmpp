#pragma once

#define TILE_SIZE 256

__global__ void mergeRangesKernel(
    const unsigned int* src,
    unsigned int* dst,
    unsigned int n,
    unsigned int width
);
