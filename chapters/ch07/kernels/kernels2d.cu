#define INIT_CONSTANT_MEMORY_2D

#include "kernels2d.cuh"
#include <cuda_runtime.h>

cudaError_t uploadConstFilter2D(const float* host, std::size_t numBytes) {
    return cudaMemcpyToSymbol(constFilter2D, host, numBytes);
}

__global__ void conv2dKernel(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width
) {
    int filterSize { 2 * radius + 1 };
    int outRow { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int outCol { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };

    // Calculating output elements
    float pValue { 0.0f };
    for (int fRow { 0 }; fRow < 2*radius + 1; ++fRow ) {
        for (int fCol { 0 }; fCol < 2*radius + 1; ++fCol) {
            int inRow { outRow - radius + fRow };
            int inCol { outCol - radius + fCol };

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                pValue += filter[fRow*filterSize + fCol] * inArray[inRow*width + inCol];
        }
    }

    if (outRow < height && outCol < width)
        outArray[outRow * width + outCol] = pValue;
}

__global__ void conv2dKernelConstMem(
    const float *inArray,
    float *outArray,
    int radius,
    int height,
    int width
) {
    int outRow { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int outCol { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };

    // Calculating output elements
    float pValue { 0.0f };
    for (int fRow { 0 }; fRow < 2*radius + 1; ++fRow ) {
        for (int fCol { 0 }; fCol < 2*radius + 1; ++fCol) {
            int inRow { outRow - radius + fRow };
            int inCol { outCol - radius + fCol };

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                pValue += constFilter2D[fRow][fCol] * inArray[inRow*width + inCol];
        }
    }

    if (outRow < height && outCol < width)
        outArray[outRow * width + outCol] = pValue;
}

// Kernel using thread organisation with thread blocks of IN_TILE_DIM size
__global__ void conv2dKernelTiledIn(
    const float *inArray,
    float *outArray,
    int radius,
    int height,
    int width
) {
    int row { static_cast<int>(blockIdx.y*config2d::OUT_TILE_DIM + threadIdx.y - radius) };
    int col { static_cast<int>(blockIdx.x*config2d::OUT_TILE_DIM + threadIdx.x - radius) };

    // Shared tile including halo
    __shared__ float inArray_s[config2d::IN_TILE_DIM][config2d::IN_TILE_DIM];

    if (row >= 0 && row < height && col >= 0 && col < width) {
        inArray_s[threadIdx.y][threadIdx.x] = inArray[row*width + col];
    } else {
        inArray_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculating output elements
    int tileRow { static_cast<int>(threadIdx.y - radius) };
    int tileCol { static_cast<int>(threadIdx.x - radius) };

    if (row >= 0 && row < height && col >= 0 && col < width) {
        if (
            tileCol >= 0 && tileCol < config2d::OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < config2d::OUT_TILE_DIM
        ) {
            float pValue { 0.0f };
            for (int fRow { 0 }; fRow < 2*radius + 1; fRow++) {
                for (int fCol { 0 }; fCol < 2*radius + 1; fCol++) {
                    pValue += constFilter2D[fRow][fCol] * inArray_s[tileRow + fRow][tileCol + fCol];
                }
            }
            outArray[row*width + col] = pValue;
        }
    }
}

// Kernel using thread organisation with thread blocks of OUT_TILE_DIM size
__global__ void conv2dKernelTiledOut(
    const float *inArray,
    float *outArray,
    int radius,
    int height,
    int width
) {
    int outRow { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int outCol { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };

    // Shared tile including halo
    __shared__ float inArray_s[config2d::IN_TILE_DIM][config2d::IN_TILE_DIM];

    // Cooperative loading IN_TILE_DIMxIN_TILE_DIM
    for (int i { static_cast<int>(threadIdx.y) }; i < config2d::IN_TILE_DIM; i += config2d::OUT_TILE_DIM) {
        for (int j { static_cast<int>(threadIdx.x) }; j < config2d::IN_TILE_DIM; j += config2d::OUT_TILE_DIM) {
            int loadRow { static_cast<int>(blockIdx.y*blockDim.y) - radius + i };
            int loadCol { static_cast<int>(blockIdx.x*blockDim.x) - radius + j };

            if (loadRow >= 0 && loadRow < height && loadCol >= 0 && loadCol < width) {
                inArray_s[i][j] = inArray[loadRow*width + loadCol];
            } else {
                inArray_s[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Calculating output elements
    float pValue { 0.0f };
    for (int fRow { 0 }; fRow < 2*radius + 1; ++fRow) {
        for (int fCol {0 }; fCol < 2*radius + 1; ++fCol) {
            int inRow { static_cast<int>(fRow + threadIdx.y) };
            int inCol { static_cast<int>(fCol + threadIdx.x) };

            if (
                inRow >= 0 && inRow < config2d::IN_TILE_DIM &&
                inCol >= 0 && inCol < config2d::IN_TILE_DIM
            )
                pValue += constFilter2D[fRow][fCol] * inArray_s[inRow][inCol]; 
        }
    }

    if (outRow < height && outCol < width)
        outArray[outRow*width + outCol] = pValue;
}


__global__ void conv2dKernelTiledCached(
    const float *inArray,
    float *outArray,
    int radius,
    int height,
    int width
) {
    int row { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int col { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };

    // Load tile to the shared memory
    __shared__ float inArrays_s[config2d::TILE_DIM][config2d::TILE_DIM];
    if (row < height && col < width) {
        inArrays_s[threadIdx.y][threadIdx.x] = inArray[row*width + col];
    } else {
        inArrays_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Computing the output elements
    if (row < height && col < width) {
        float pValue { 0.0f };

        for (int fRow { 0 }; fRow < 2*radius + 1; fRow++) {
            for (int fCol { 0 }; fCol < 2*radius + 1; fCol++) {
                if (
                    threadIdx.x - radius + fCol < config2d::TILE_DIM &&
                    threadIdx.y - radius + fRow < config2d::TILE_DIM
                ) {
                    pValue += constFilter2D[fRow][fCol] * inArrays_s[
                        threadIdx.y - radius + fRow][threadIdx.x - radius + fCol];
                } else {
                    if (
                        row - radius + fRow >= 0 &&
                        row - radius + fRow < height &&
                        col - radius + fCol >= 0 &&
                        col - radius + fCol < width
                    ) {
                        pValue += constFilter2D[fRow][fCol] * inArray[
                            (row - radius + fRow)*width + col - radius + fCol];
                    }
                }
            }
        }
        if (row < height && col < width)
            outArray[row*width + col] = pValue;
    }
}
