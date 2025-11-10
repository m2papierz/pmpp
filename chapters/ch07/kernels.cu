#define INIT_CONSTANT_MEMORY

#include "kernels.cuh"
#include <cuda_runtime.h>

cudaError_t uploadConstFilter(const float* host, std::size_t numBytes) {
    return cudaMemcpyToSymbol(constFilter, host, numBytes);
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

    float pValue { 0.0f };
    for (int fRow { 0 }; fRow < 2*radius + 1; ++fRow ) {
        for (int fCol { 0 }; fCol < 2*radius + 1; ++fCol) {
            int inRow { outRow - radius + fRow };
            int inCol { outCol - radius + fCol };

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                pValue += filter[fRow*filterSize + fCol] * inArray[inRow*width + inCol];
        }
    }

    if (outRow >= 0 && outRow < height && outCol >= 0 && outCol < width)
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

    float pValue { 0.0f };
    for (int fRow { 0 }; fRow < 2*radius + 1; ++fRow ) {
        for (int fCol { 0 }; fCol < 2*radius + 1; ++fCol) {
            int inRow { outRow - radius + fRow };
            int inCol { outCol - radius + fCol };

            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                pValue += constFilter[fRow][fCol] * inArray[inRow*width + inCol];
        }
    }
    if (outRow >= 0 && outRow < height && outCol >= 0 && outCol < width)
        outArray[outRow * width + outCol] = pValue;
}
