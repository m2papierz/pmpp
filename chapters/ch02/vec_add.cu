#include "utils.hpp"
#include <vector>
#include <iostream>

#define BLOCK_SIZE 32

// vector addition in plain C
void vecAddCPU(const float* A_h, const float* B_h, float* C_h, int n) {
    for (int i  { 0 }; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

// CUDA kernel for vector addition
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i { static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x) };
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// vector addition in CUDA C
void vecAddCUDA(const float* A_h, const float* B_h, float* C_h, int n) {
    std::size_t size { static_cast<std::size_t>(n) * sizeof(float) };
    
    // Allocate device memory for A, B, and C
    float *A_d { nullptr }, *B_d { nullptr }, *C_d { nullptr };
    CUDA_CHECK(cudaMalloc(&A_d, size));
    CUDA_CHECK(cudaMalloc(&B_d, size));
    CUDA_CHECK(cudaMalloc(&C_d, size));

    // Copy A, and B to device memory
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Call kernel to launch a grid of threads
    vecAddKernel<<<utils::cdiv(n, BLOCK_SIZE), BLOCK_SIZE>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy C from the device memory
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main(){
    const int N { 1<<24 }; // ~16 million elements

    // Alocate input vectors in host memory
    std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N);

    // Init inputs
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    // Run the C version and time it
    double secondsCPU = utils::executeAndTimeFunction([&]{
        vecAddCPU(A.data(), B.data(), C_cpu.data(), N);
    });
    std::cout << "CPU version elapsed time: " << secondsCPU << "seconds\n";

    // Run the CUDA version and time it
    double secondsGPU = utils::cudaExecuteAndTimeFunction([&]{
        vecAddCUDA(A.data(), B.data(), C_gpu.data(), N);
    });
    std::cout << "GPU version elapsed time: " << secondsGPU << "seconds\n";

    return 0;
}
