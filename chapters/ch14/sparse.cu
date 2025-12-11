#include "sparse.cuh"

void spmv_coo_cpu(
    const COOMatrix& cooMatrix,
    const float* x,
    float* y
) {
    std::fill(y, y + cooMatrix.rows, 0.0f);

    for (std::size_t i = 0; i < cooMatrix.numNonzeros; ++i) {
        int row = cooMatrix.rowIdx[i];
        int col = cooMatrix.colIdx[i];
        y[row] += cooMatrix.values[i]*x[col];
    }
}

void spmv_coo(
    const COOMatrix& cooMatrix,
    const float* x,
    float* y
) {
    COOMatrixDevice cooMatrix_d { cooMatrix };

    float *x_d { nullptr }, *y_d { nullptr };
    CUDA_CHECK(cudaMalloc(&x_d, cooMatrix.cols*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, cooMatrix.rows*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(x_d, x, cooMatrix.cols*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(y_d, 0, cooMatrix.rows*sizeof(float)));

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(cooMatrix.numNonzeros, BLOCK_SIZE));
    spmv_coo_kernel<<<gridSize, blockSize>>>(cooMatrix_d.dev, x_d, y_d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(y, y_d, cooMatrix.rows*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
}
