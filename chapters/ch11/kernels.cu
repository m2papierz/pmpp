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

__global__ void coarsenedThreePhaseKernel(const float* x, float* y, unsigned int n) {
    extern __shared__ float shared_mem[];
    float* buffer = shared_mem;
    float* last_elements = &shared_mem[n];

    // Phase 1 - coarsened loading to the shared memory
    for (unsigned int i { 0 }; i < COARSE_FACTOR; ++i) {
        unsigned int idx { threadIdx.x*COARSE_FACTOR + i };
        if (idx < n) {
            buffer[idx] = x[idx]; 
        }
    }
    __syncthreads();

    // Coarsened local scan
    for (unsigned int i { 1 }; i < COARSE_FACTOR; ++i) {
        unsigned int idx { threadIdx.x*COARSE_FACTOR + i };
        if (idx < n) {
            buffer[idx] += buffer[idx - 1];
        }
    }
    __syncthreads();

    // Load positions of last elements in the section
    if (threadIdx.x < n / COARSE_FACTOR) {
        unsigned int end_idx { (threadIdx.x + 1)*COARSE_FACTOR - 1 };
        if (end_idx < n) {
            last_elements[threadIdx.x] = buffer[end_idx];
        }
    }

    // Phase 2 - apply Kogge-Stone scan on the ends of each section
    unsigned int num_sections { (n + COARSE_FACTOR - 1) / COARSE_FACTOR };

    for (int stride { 1 }; stride < num_sections; stride *= 2) {
        __syncthreads();

        float temp{};
        if (threadIdx.x >= stride && threadIdx.x < num_sections) {
            temp = last_elements[threadIdx.x] + last_elements[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride && threadIdx.x < num_sections) {
            last_elements[threadIdx.x] = temp;
        }
    }
    __syncthreads();

    // Phase 3 - final section sums
    for (unsigned int i { 0 }; i < COARSE_FACTOR; ++i) {
        unsigned int idx { threadIdx.x*COARSE_FACTOR + i};
        if (idx < n) {
            unsigned int section_idx { idx / COARSE_FACTOR };
            if (section_idx > 0) {
                buffer[idx] += last_elements[section_idx - 1];
            }
            y[idx] = buffer[idx];
        }
    }
}
