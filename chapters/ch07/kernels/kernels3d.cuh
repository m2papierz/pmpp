#pragma once
#include <cuda_runtime.h>

namespace config3d {
    inline constexpr int FILTER_RADIUS { 3 };
    inline constexpr int BLOCK_SIZE { 8 };                                       // For basic implementation
    inline constexpr int IN_TILE_DIM { 8 };                                      // For tiled implementation
    inline constexpr int OUT_TILE_DIM { ((IN_TILE_DIM) - 2*(FILTER_RADIUS)) };    // For tiled implementation
    inline constexpr int TILE_DIM { 8 };
    inline constexpr int FILTER_DIM { 2 * FILTER_RADIUS + 1 };
}

#ifdef INIT_CONSTANT_MEMORY_3D
__constant__ float constFilter3D[config3d::FILTER_DIM][config3d::FILTER_DIM][config3d::FILTER_DIM];
#else
extern __constant__ float constFilter3D[config3d::FILTER_DIM][config3d::FILTER_DIM][config3d::FILTER_DIM];
#endif

cudaError_t uploadConstFilter3D(const float* host, std::size_t numBytes);
__global__ void conv3dKernel(const float *inArray, const float *filter, float *outArray, int radius, int height, int width, int depth);
__global__ void conv3dKernelConstMem(const float *inArray, float *outArray, int radius, int height, int width, int depth);
__global__ void conv3dKernelTiled(const float *inArray, float *outArray, int radius, int height, int width, int depth);
