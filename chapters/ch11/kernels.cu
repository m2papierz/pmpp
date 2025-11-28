#include "kernels.cuh"
#include <cuda_runtime.h>

__global__ void koggeStoneKernel(const float* x, float* y, unsigned int n) {
    unsigned int i { blockIdx.x*blockDim.x + threadIdx.x };
    
    __shared__ float xy[SECTION_SIZE];
    if (i < n) {
        xy[threadIdx.x] = x[i];
    } else {
        xy[threadIdx.x] = 0.0f;
    }

    for (int stride { 1 }; stride < blockDim.x; stride *= 2) {
        __syncthreads();

        float temp{};
        if (threadIdx.x >= stride) {
            temp = xy[threadIdx.x] + xy[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            xy[threadIdx.x] = temp;
        }
    }    

    if (i < n) {
        y[i] = xy[threadIdx.x];
    }
}

__global__ void koggeStoneKernelDoubleBuffer(const float* x, float* y, unsigned int n) {
    unsigned int i { blockIdx.x*blockDim.x + threadIdx.x };
    
    __shared__ float xy_src[SECTION_SIZE];
    __shared__ float xy_trg[SECTION_SIZE];

    float* src { xy_src };
    float* trg { xy_trg };

    if (i < n) {
        src[threadIdx.x] = x[i];
    } else {
        src[threadIdx.x] = 0.0f;
    }

    for (int stride { 1 }; stride < blockDim.x; stride *=2) {
        __syncthreads();
        if (threadIdx.x >= stride) {
            trg[threadIdx.x] = src[threadIdx.x] + src[threadIdx.x - stride];
        } else {
            trg[threadIdx.x] = src[threadIdx.x];
        }

        float* tmp { src };
        src = trg;
        trg = tmp;

    }

    if (i < n) {
        y[i] = src[threadIdx.x];
    }
}


__global__ void brentKungKernel(const float* x, float* y, unsigned int n) {
    unsigned int i { 2*blockDim.x*blockIdx.x + threadIdx.x };
    
    __shared__ float xy[SECTION_SIZE];
    if (i < n) {
        xy[threadIdx.x] = x[i];
    }

    if (i + blockDim.x < n) {
        xy[threadIdx.x + blockDim.x] = x[i + blockDim.x];
    }

    for (unsigned int stride { 1 }; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index { (threadIdx.x + 1)*2*stride - 1 };
        if (index < SECTION_SIZE) {
            xy[index] += xy[index - stride];
        }
    }

    for (unsigned int stride { SECTION_SIZE/4 }; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index { (threadIdx.x + 1)*2*stride - 1 };
        if (index + stride < SECTION_SIZE) {
            xy[index + stride] += xy[index];
        }
    }
    __syncthreads();

    if (i < n) {
        y[i] = xy[threadIdx.x];
    }

    if (i + blockDim.x < n) {
        y[i + blockDim.x] = xy[threadIdx.x + blockDim.x];
    }
}
