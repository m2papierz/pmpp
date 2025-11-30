#include "merge.cuh"
#include "kernels.cuh"
#include "utils.hpp"

void mergeBasic(
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C
) {
    std::size_t sizeA { static_cast<std::size_t>(m*sizeof(int)) };
    std::size_t sizeB { static_cast<std::size_t>(n*sizeof(int)) };
    std::size_t sizeC { static_cast<std::size_t>((m + n)*sizeof(int)) };

    int *A_d { nullptr }, *B_d { nullptr }, *C_d { nullptr };
    CUDA_CHECK(cudaMalloc(&A_d, sizeA));
    CUDA_CHECK(cudaMalloc(&B_d, sizeB));
    CUDA_CHECK(cudaMalloc(&C_d, sizeC));

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(m + n, BLOCK_SIZE));
    mergeBasicKernel<<<gridSize, blockSize>>>(A_d, m, B_d, n, C_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

void mergeTiled(
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C
) {
    std::size_t sizeA { static_cast<std::size_t>(m*sizeof(int)) };
    std::size_t sizeB { static_cast<std::size_t>(n*sizeof(int)) };
    std::size_t sizeC { static_cast<std::size_t>((m + n)*sizeof(int)) };
    std::size_t sharedMemSize { 2*TILE_SIZE*sizeof(int) };

    int *A_d { nullptr }, *B_d { nullptr }, *C_d { nullptr };
    CUDA_CHECK(cudaMalloc(&A_d, sizeA));
    CUDA_CHECK(cudaMalloc(&B_d, sizeB));
    CUDA_CHECK(cudaMalloc(&C_d, sizeC));

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(m + n, BLOCK_SIZE));
    mergeTiledKernel<<<gridSize, blockSize, sharedMemSize>>>(A_d, m, B_d, n, C_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

void mergeCircularBuffer(
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C
) {
    std::size_t sizeA { static_cast<std::size_t>(m*sizeof(int)) };
    std::size_t sizeB { static_cast<std::size_t>(n*sizeof(int)) };
    std::size_t sizeC { static_cast<std::size_t>((m + n)*sizeof(int)) };
    std::size_t sharedMemSize { 2*TILE_SIZE*sizeof(int) };

    int *A_d { nullptr }, *B_d { nullptr }, *C_d { nullptr };
    CUDA_CHECK(cudaMalloc(&A_d, sizeA));
    CUDA_CHECK(cudaMalloc(&B_d, sizeB));
    CUDA_CHECK(cudaMalloc(&C_d, sizeC));

    CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(m + n, BLOCK_SIZE));
    mergeCircularBufferKernel<<<gridSize, blockSize, sharedMemSize>>>(A_d, m, B_d, n, C_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}
