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

__global__ void spmv_crs_kernel(
    DeviceCRSMatrix crsMatrix,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    unsigned int row { blockIdx.x*blockDim.x + threadIdx.x };
    if (row < crsMatrix.numRows) {
        float sum { 0.0f };
        for (int i { crsMatrix.rowPtrs[row] }; i < crsMatrix.rowPtrs[row +1]; ++i) {
            int col = crsMatrix.colIdx[i];
            float value = crsMatrix.values[i];
            sum += x[col]*value;
        }
        y[row] += sum;
    }
}
