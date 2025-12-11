#include "kernels.cuh"

__global__ void spmv_coo_kernel(
    DeviceCOOMatrix cooMatrix,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    unsigned int i { blockIdx.x*blockDim.x + threadIdx.x };

    if (i < cooMatrix.numNonzeros) {
        int row { cooMatrix.rowIdx[i] };
        int col { cooMatrix.colIdx[i] };
        float value { cooMatrix.values[i] };
        atomicAdd(&y[row], x[col]*value);
    }
}
