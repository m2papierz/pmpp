#include "formats.cuh"

COOMatrixDevice::COOMatrixDevice(const COOMatrix& h) {
    dev.numNonzeros = static_cast<int>(h.numNonzeros);
    if (dev.numNonzeros == 0) return;

    CUDA_CHECK(cudaMalloc(&d_rowIdx,  dev.numNonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx,  dev.numNonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values,  dev.numNonzeros * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rowIdx, h.rowIdx.data(), dev.numNonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h.colIdx.data(), dev.numNonzeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h.values.data(), dev.numNonzeros * sizeof(float), cudaMemcpyHostToDevice));

    dev.rowIdx = d_rowIdx;
    dev.colIdx = d_colIdx;
    dev.values = d_values;
}

COOMatrixDevice::~COOMatrixDevice() {
    if (d_rowIdx)  cudaFree(d_rowIdx);
    if (d_colIdx)  cudaFree(d_colIdx);
    if (d_values)  cudaFree(d_values);
}

COOMatrixDevice::COOMatrixDevice(COOMatrixDevice&& other) noexcept {
    *this = std::move(other);
}

COOMatrixDevice& COOMatrixDevice::operator=(COOMatrixDevice&& other) noexcept {
    if (this != &other) {
        if (d_rowIdx)  cudaFree(d_rowIdx);
        if (d_colIdx)  cudaFree(d_colIdx);
        if (d_values)  cudaFree(d_values);

        dev = other.dev;
        d_rowIdx = other.d_rowIdx;
        d_colIdx = other.d_colIdx;
        d_values = other.d_values;

        other.dev = {};
        other.d_rowIdx = nullptr;
        other.d_colIdx = nullptr;
        other.d_values = nullptr;
    }
    return *this;
}
