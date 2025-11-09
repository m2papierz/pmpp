#include "utils.hpp"
#include <iostream>

#define THREADS_PER_BLOCK 32

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
    unsigned int col { blockIdx.x * blockDim.x + threadIdx.x };
    unsigned int row { blockIdx.y * blockDim.y + threadIdx.y };
    if (row < n && col < k) {
        float pValue = 0;
        for (int i { 0 }; i < m; ++i) {
            pValue += A[row * m + i] * B[i * k + col];
        }
        C[row * k + col] = pValue;
    }
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
    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    dim3 gridSize(
        (k + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK
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
    std::vector<float> C_cpu(Constants::n * Constants::k);
    std::vector<float> C_gpu(Constants::n * Constants::k);

    // Initialize matrices with random values
    for (int i { 0 }; i < Constants::n * Constants::m; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i { 0 }; i < Constants::m * Constants::k; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform matmul in C and time it
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            utils::matMulCPU(
                A.data(), B.data(), C_cpu.data(),
                Constants::n, Constants::m, Constants::k
            );
        })
    };
    std::cout << "CPU version elapsed time: " << secondsCPU << "seconds\n";

    // Perform matmul in CUDA and time it
    float secondsGPU {
        utils::cudaExecuteAndTimeFunction([&]{
            matMulCUDA(
                A.data(), B.data(), C_gpu.data(),
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
