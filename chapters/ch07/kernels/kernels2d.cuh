#pragma once
#include <cuda_runtime.h>                                      // For L2 cache implementation

namespace config2d {
    inline constexpr int FILTER_RADIUS { 3 };
    inline constexpr int BLOCK_SIZE { 32 };                                       // For basic implementation
    inline constexpr int IN_TILE_DIM { 32 };                                      // For tiled implementation
    inline constexpr int OUT_TILE_DIM { ((IN_TILE_DIM) - 2*(FILTER_RADIUS)) };    // For tiled implementation
    inline constexpr int TILE_DIM { 32 }; 
    inline constexpr int FILTER_DIM { 2 * FILTER_RADIUS + 1 };
}

#ifdef INIT_CONSTANT_MEMORY_2D
__constant__ float constFilter2D[config2d::FILTER_DIM][config2d::FILTER_DIM];
#else
extern __constant__ float constFilter2D[config2d::FILTER_DIM][config2d::FILTER_DIM];
#endif

cudaError_t uploadConstFilter2D(const float* host, std::size_t numBytes);
__global__ void conv2dKernel(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
__global__ void conv2dKernelConstMem(const float *inArray, float *outArray, int radius, int height, int width);
__global__ void conv2dKernelTiledIn(const float *inArray, float *outArray, int radius, int height, int width);
__global__ void conv2dKernelTiledOut(const float *inArray, float *outArray, int radius, int height, int width);
__global__ void conv2dKernelTiledCached(const float *inArray, float *outArray, int radius, int height, int width);
