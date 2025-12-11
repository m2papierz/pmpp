#include "formats.cuh"

// ---------- COO format ----------

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

// ---------- CRS format ----------

CRSMatrixDevice::CRSMatrixDevice(const CRSMatrix& h) {
    dev.numRows = h.numRows;
    dev.numNonzeros = static_cast<int>(h.numNonzeros);

    if (dev.numRows == 0 || dev.numNonzeros == 0) return;

    CUDA_CHECK(cudaMalloc(&d_rowPtrs, (h.numRows + 1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_colIdx, dev.numNonzeros*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, dev.numNonzeros*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rowPtrs, h.rowPtrs.data(), (h.numRows + 1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, h.colIdx.data(), dev.numNonzeros*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h.values.data(), dev.numNonzeros*sizeof(float), cudaMemcpyHostToDevice));

    dev.rowPtrs = d_rowPtrs;
    dev.colIdx = d_colIdx;
    dev.values = d_values;
}

CRSMatrixDevice::~CRSMatrixDevice() {
    if (d_rowPtrs) cudaFree(d_rowPtrs);
    if (d_colIdx)  cudaFree(d_colIdx);
    if (d_values)  cudaFree(d_values);
}

CRSMatrixDevice::CRSMatrixDevice(CRSMatrixDevice&& other) noexcept {
    *this = std::move(other);
}

CRSMatrixDevice& CRSMatrixDevice::operator=(CRSMatrixDevice&& other) noexcept {
    if (this != &other) {
        if (d_rowPtrs) cudaFree(d_rowPtrs);
        if (d_colIdx) cudaFree(d_colIdx);
        if (d_values) cudaFree(d_values);

        dev = other.dev;
        d_rowPtrs = other.d_rowPtrs;
        d_colIdx  = other.d_colIdx;
        d_values  = other.d_values;

        other.dev = {};
        other.d_rowPtrs = nullptr;
        other.d_colIdx  = nullptr;
        other.d_values  = nullptr;
    }
    return *this;
}
