#define INIT_CONSTANT_MEMORY_3D

#include "kernels3d.cuh"
#include <cuda_runtime.h>

cudaError_t uploadConstFilter3D(const float* host, std::size_t numBytes) {
    return cudaMemcpyToSymbol(constFilter3D, host, numBytes);
}

__global__ void conv3dKernel(
    const float *inArray,
    const float *filter,
    float *outArray,
    int radius,
    int height,
    int width,
    int depth
) {
    int filterSize { 2*radius + 1 };
    int outRow { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int outCol { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };
    int outDep { static_cast<int>(blockIdx.z*blockDim.z + threadIdx.z) };

    // Calculating output elements
    float pValue { 0.0f };
    for (int fDep { 0 }; fDep < filterSize; ++fDep ) {
        for (int fRow { 0 }; fRow < filterSize; ++fRow ) {
            for (int fCol { 0 }; fCol < filterSize; ++fCol) {
                int inRow { outRow - radius + fRow };
                int inCol { outCol - radius + fCol };
                int inDep { outDep - radius + fDep };

                if (
                    inRow >= 0 && inRow < height &&
                    inCol >= 0 && inCol < width &&
                    inDep >= 0 && inDep < depth
                ) {
                    int inIdx { inDep*height*width + inRow*width + inCol };
                    int filterIdx { fDep*filterSize*filterSize + fRow*filterSize + fCol };
                    pValue += filter[filterIdx] * inArray[inIdx];
                }
            }
        }
    }

    if (outRow < height && outCol < width && outDep < depth)
        outArray[outDep*height*width + outRow*width + outCol] = pValue;
}

__global__ void conv3dKernelConstMem(
    const float *inArray,
    float *outArray,
    int radius,
    int height,
    int width,
    int depth
) {
    int filterSize { 2 * radius + 1 };
    int outRow { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int outCol { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };
    int outDep { static_cast<int>(blockIdx.z*blockDim.z + threadIdx.z) };

    // Calculating output elements
    float pValue { 0.0f };
    for (int fDep { 0 }; fDep < 2*radius + 1; ++fDep ) {
        for (int fRow { 0 }; fRow < filterSize; ++fRow ) {
            for (int fCol { 0 }; fCol < filterSize; ++fCol) {
                int inRow { outRow - radius + fRow };
                int inCol { outCol - radius + fCol };
                int inDep { outDep - radius + fDep };

                if (
                    inRow >= 0 && inRow < height &&
                    inCol >= 0 && inCol < width &&
                    inDep >= 0 && inDep < depth
                ) {
                    pValue += constFilter3D[fDep][fRow][fCol] * inArray[
                        inDep*height*width + inRow*width + inCol];
                }
            }
        }
    }

    if (outRow < height && outCol < width && outDep < depth)
        outArray[outDep*height*width + outRow * width + outCol] = pValue;
}

// Kernel using thread organisation with thread blocks of IN_TILE_DIM size
__global__ void conv3dKernelTiled(
    const float *inArray,
    float *outArray,
    int radius,
    int height,
    int width,
    int depth
) {
    int row { static_cast<int>(blockIdx.y*config3d::OUT_TILE_DIM + threadIdx.y - radius) };
    int col { static_cast<int>(blockIdx.x*config3d::OUT_TILE_DIM + threadIdx.x - radius) };
    int dep { static_cast<int>(blockIdx.z*config3d::OUT_TILE_DIM + threadIdx.z - radius) };

    // Shared tile including halo
    __shared__ float inArray_s[config3d::IN_TILE_DIM][config3d::IN_TILE_DIM][config3d::IN_TILE_DIM];

    if (
        row >= 0 && row < height &&
        col >= 0 && col < width &&
        dep >=0 && dep < depth
    ) {
        inArray_s[threadIdx.z][threadIdx.y][threadIdx.x] = inArray[dep*height*width + row*width + col];
    } else {
        inArray_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculating output elements
    int tileRow { static_cast<int>(threadIdx.y - radius) };
    int tileCol { static_cast<int>(threadIdx.x - radius) };
    int tileDep { static_cast<int>(threadIdx.z - radius) };

    if (
        row >= 0 && row < height &&
        col >= 0 && col < width &&
        dep >= 0 && dep < depth
    ) {
        if (
            tileCol >= 0 && tileCol < config3d::OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < config3d::OUT_TILE_DIM &&
            tileDep >= 0 && tileDep < config3d::OUT_TILE_DIM
        ) {
            float pValue { 0.0f };
            for (int fDep { 0 }; fDep < 2*radius + 1; fDep++) {
                for (int fRow { 0 }; fRow < 2*radius + 1; fRow++) {
                    for (int fCol { 0 }; fCol < 2*radius + 1; fCol++) {
                        pValue += constFilter3D[fDep][fRow][fCol] * inArray_s[
                            tileDep + fDep][tileRow + fRow][tileCol + fCol];
                    }
                }
            }
            outArray[dep*height*width + row*width + col] = pValue;
        }
    }
}
