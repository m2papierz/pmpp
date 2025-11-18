#include "histogram.cuh"
#include "utils.hpp"
#include "kernels.cuh"

void histogramSequential(
    const char* data,
    unsigned int length,
    unsigned int* histo
) {
    for (unsigned int i { 0 }; i < length; ++i) {
        int alphabetPosition { data[i] - 'a' };
        if (alphabetPosition >= 0 && alphabetPosition < 26) {
            histo[alphabetPosition/BIN_SIZE]++;
        }
    }
}

void histogramNaive(
    const char* data,
    unsigned int length,
    unsigned int* histo
) {
    std::size_t sizeData { static_cast<std::size_t>(length*sizeof(char)) };
    std::size_t sizeOut { static_cast<std::size_t>(NUM_BINS*sizeof(unsigned int)) };

    char* data_d { nullptr };
    CUDA_CHECK(cudaMalloc(&data_d, sizeData));
    CUDA_CHECK(cudaMemcpy(data_d, data, sizeData, cudaMemcpyHostToDevice));

    unsigned int* histo_d { nullptr };
    CUDA_CHECK(cudaMalloc(&histo_d, sizeOut));
    CUDA_CHECK(cudaMemcpy(histo_d, histo, sizeOut, cudaMemcpyHostToDevice));

    histogramKernel<<<utils::cdiv(length, BLOCK_SIZE), BLOCK_SIZE>>>(data_d, length, histo_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(histo, histo_d, sizeOut, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(histo_d));
}

void histogramPrivate(
    const char* data,
    unsigned int length,
    unsigned int* histo
) {
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(utils::cdiv(length, BLOCK_SIZE));

    std::size_t sizeData { static_cast<std::size_t>(length*sizeof(char)) };
    std::size_t sizeOut_d { static_cast<std::size_t>(NUM_BINS*gridDim.x*sizeof(unsigned int)) };
    std::size_t sizeOut_h { static_cast<std::size_t>(NUM_BINS*sizeof(unsigned int)) };

    char* data_d { nullptr };
    CUDA_CHECK(cudaMalloc(&data_d, sizeData));
    CUDA_CHECK(cudaMemcpy(data_d, data, sizeData, cudaMemcpyHostToDevice));

    unsigned int* histo_d { nullptr };
    CUDA_CHECK(cudaMalloc(&histo_d, sizeOut_d));

    histogramKernelPrivate<<<gridDim, blockDim>>>(data_d, length, histo_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(histo, histo_d, sizeOut_h, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(histo_d));
}


void histogramSharedMem(
    const char* data,
    unsigned int length,
    unsigned int* histo
) {
    std::size_t sizeData { static_cast<std::size_t>(length*sizeof(char)) };
    std::size_t sizeOut { static_cast<std::size_t>(NUM_BINS*sizeof(unsigned int)) };

    char* data_d { nullptr };
    CUDA_CHECK(cudaMalloc(&data_d, sizeData));
    CUDA_CHECK(cudaMemcpy(data_d, data, sizeData, cudaMemcpyHostToDevice));

    unsigned int* histo_d { nullptr };
    CUDA_CHECK(cudaMalloc(&histo_d, sizeOut));
    CUDA_CHECK(cudaMemcpy(histo_d, histo, sizeOut, cudaMemcpyHostToDevice));

    histogramKernelSharedMem<<<utils::cdiv(length, BLOCK_SIZE), BLOCK_SIZE>>>(data_d, length, histo_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(histo, histo_d, sizeOut, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(histo_d));
}
