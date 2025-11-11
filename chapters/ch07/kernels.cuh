#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define FILTER_RADIUS 3
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

#ifdef INIT_CONSTANT_MEMORY
__constant__ float constFilter[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];
#else
extern __constant__ float constFilter[2*FILTER_RADIUS + 1][2*FILTER_RADIUS + 1];
#endif

cudaError_t uploadConstFilter(const float* host, std::size_t numBytes);
__global__ void conv2dKernel(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
__global__ void conv2dKernelConstMem(const float *inArray, float *outArray, int radius, int height, int width);
__global__ void conv2dKernelTiledIn(const float *inArray, float *outArray, int radius, int height, int width);
__global__ void conv2dKernelTiledOut(const float *inArray, float *outArray, int radius, int height, int width);
