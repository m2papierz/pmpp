#include "utils.hpp"
#include <vector>
#include <iostream>

#define TILE_WIDTH 32

namespace Constants {
    constexpr int n { 2048 };
    constexpr int m { 1024 };
    constexpr int k { 2048 }; 
}

__global__ void matMulKernel(
    const float* A,
    const float* B,
    float* C,
    int n, int m, int k
) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    unsigned int bx { blockIdx.x };
    unsigned int by { blockIdx.y };
    unsigned int tx { threadIdx.x };
    unsigned int ty { threadIdx.y };

    // Identify the row and column of the C element to work on
    unsigned int row { by * TILE_WIDTH + ty };
    unsigned int col { bx * TILE_WIDTH + tx };

    // Loop over the A and B tiles required to compute C element
    float pValue { 0 };
    int numTiles { (m + TILE_WIDTH - 1) / TILE_WIDTH };
    for (int ph = 0; ph < numTiles; ++ph) {

        // Loading of A and B tiles into shared memory
        if ((row < n) && (ph*TILE_WIDTH + tx) < m)
            Mds[ty][tx] = A[row*m + ph*TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if ((ph*TILE_WIDTH + ty) < m && (col < k))
            Nds[ty][tx] = B[col*m + ph*TILE_WIDTH + ty];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            pValue += Mds[ty][i] * Nds[i][tx];
        }
        __syncthreads();

    }

    if (row < n && col < k)
        C[row*k + col] = pValue;
}

void matMulCUDA(
    const float* A,
    const float* B,
    float* C,
    int n, int m, int k
) {
    // Allocate device memory
    std::size_t sizeA { (std::size_t)(n * m * sizeof(float)) };
    std::size_t sizeB { (std::size_t)(m * k * sizeof(float)) };
    std::size_t sizeC { (std::size_t)(n * k * sizeof(float)) };

    float *A_d { nullptr }, *B_d { nullptr }, *C_d { nullptr };
    CUDA_CHECK(cudaMalloc((void**)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&C_d, sizeC));

    // copy matrices to device
    CUDA_CHECK(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridSize(
        (k + TILE_WIDTH - 1) / TILE_WIDTH,
        (n + TILE_WIDTH - 1) / TILE_WIDTH
    );
    matMulKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, n, m, k);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main() {
    std::vector<float> A(Constants::n * Constants::m);
    std::vector<float> B(Constants::m * Constants::k);
    std::vector<float> B_t(Constants::m * Constants::k);
    std::vector<float> C_cpu(Constants::n * Constants::k);
    std::vector<float> C_gpu(Constants::n * Constants::k);

    // Initialize matrices with random values
    for (int i = 0; i < Constants::n * Constants::m; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < Constants::m * Constants::k; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Transpose the B matrix so that it is column-major
    utils::transposeMatrixTiled(B, B_t, Constants::m, Constants::k);

    // Perform matmul in C and time it
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            utils::matMulCPU(
                A.data(), B.data(), C_cpu.data(),
                Constants::n, Constants::m ,Constants::k
            );
        })
    };
    std::cout << "CPU version elapsed time: " << secondsCPU << "seconds\n";

    // Perform matmul in CUDA and time it
    float secondsGPU {
        utils::cudaExecuteAndTimeFunction([&]{
            matMulCUDA(
                A.data(), B_t.data(), C_gpu.data(),
                Constants::n, Constants::m, Constants::k
            );
        })
    };
    std::cout << "GPU version elapsed time: " << secondsGPU << "seconds\n";

    // Check if results are the same
    bool ok { utils::matricesAlmostEqual(
        C_cpu, C_gpu, Constants::n, Constants::k, 1e-6f, 1e-6f)
    };
    std::cout << (ok ? "OK\n" : "MISMATCH!\n");

    return 0;
}
