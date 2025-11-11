#include "utils.hpp"
#include <cmath>
#include <iostream>

namespace Constants {
    constexpr int n { 2048 };
    constexpr int m { 1024 };
    constexpr int k { 2048 }; 
}

int calculateOptimalTileWidth(int n, int m, int k) {
    // Get hardware properties
    cudaDeviceProp deviceProp {};
    cudaGetDeviceProperties(&deviceProp, 0);

    // Get hardware limits
    int maxBlockDimX { deviceProp.maxThreadsDim[0] };
    int maxBlockDimY { deviceProp.maxThreadsDim[1] };
    int maxThreadsPerBlock { deviceProp.maxThreadsPerBlock };
    int sharedMemPerBlock { static_cast<int>(deviceProp.sharedMemPerBlock) };
    int maxTileWidthBySharedMem {
        static_cast<int>(sqrt(sharedMemPerBlock / (2*sizeof(float))))
    };

    // Calculate max possible tile size based on hardware constraints
    int tileWidth { static_cast<int>(std::sqrt(maxThreadsPerBlock)) };      // based on max threads per block
    tileWidth = std::min(tileWidth, std::min(maxBlockDimX, maxBlockDimY));  // based on max block dims
    tileWidth = std::min(tileWidth, maxTileWidthBySharedMem);               // based on max shared mem
    tileWidth = std::min(tileWidth, std::min(n, std::min(m, k)));           // based on matrices dims
    tileWidth = 1 << static_cast<int>(std::log2(tileWidth));                // power of 2 for mem alignment
    tileWidth = std::max(16, tileWidth);                                    // ensure min practical size

    return tileWidth;
}

__global__ void matMulKernel(
    const float* A,
    const float* B,
    float* C,
    int n, int m, int k,
    int tileWidth
) {
    extern __shared__ float sharedMem[];
    float *Mds { sharedMem };
    float *Nds { &sharedMem[tileWidth * tileWidth]};

    int bx { static_cast<int>(blockIdx.x) };
    int by { static_cast<int>(blockIdx.y) };
    int tx { static_cast<int>(threadIdx.x) };
    int ty { static_cast<int>(threadIdx.y) };

    // Identify the row and column of the C element to work on
    int row { by * tileWidth + ty };
    int col { bx * tileWidth + tx };

    // Loop over the A and B tiles required to compute C element
    float pValue { 0 };
    int numTiles { (m + tileWidth - 1) / tileWidth };
    for (int ph { 0 }; ph < numTiles; ++ph) {

        // Loading of A and B tiles into shared memory
        if ((row < n) && (ph*tileWidth + tx) < m)
            Mds[ty*tileWidth + tx] = A[row*m + ph*tileWidth + tx];
        else
            Mds[ty*tileWidth + tx] = 0.0f;

        if ((ph*tileWidth + ty) < m && (col < k))
            Nds[ty*tileWidth + tx] = B[(ph*tileWidth + ty)*k + col];
        else
            Nds[ty*tileWidth + tx] = 0.0f;

        __syncthreads();

        for (int i  { 0 }; i < tileWidth; i++) {
            pValue += Mds[ty*tileWidth + i] * Nds[i*tileWidth + tx];
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
    int n, int m, int k,
    int tileWidth
) {
    std::size_t sizeM { static_cast<std::size_t>(n * m * sizeof(float)) };
    std::size_t sizeN { static_cast<std::size_t>(m * k * sizeof(float)) };
    std::size_t sizeP { static_cast<std::size_t>(n * k * sizeof(float)) };

    // Allocate device memory
    float *M_d { nullptr }, *N_d { nullptr }, *P_d { nullptr };
    CUDA_CHECK(cudaMalloc(&M_d, sizeM));
    CUDA_CHECK(cudaMalloc(&N_d, sizeN));
    CUDA_CHECK(cudaMalloc(&P_d, sizeP));

    // Copy data from host do device
    CUDA_CHECK(cudaMemcpy(M_d, A, sizeM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, B, sizeN, cudaMemcpyHostToDevice));

    // Call the kernel
    dim3 blockSize(tileWidth, tileWidth, 1);
    dim3 gridSize(utils::cdiv(k, tileWidth), utils::cdiv(n, tileWidth));
    std::size_t sharedMemSize { 2 * tileWidth * tileWidth * sizeof(float) };
    matMulKernel<<<gridSize, blockSize, sharedMemSize>>>(M_d, N_d, P_d, n, m, k, tileWidth);

    // Check launch/runtime errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(C, P_d, sizeP, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));
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

    // Compute optimal tile width for hardware
    int tileWidth { calculateOptimalTileWidth(Constants::n, Constants::m, Constants::k) };
    std::cout << "Calculated optimal tile width: " << tileWidth << std::endl;

    // Perform matmul in CUDA and time it
    float secondsGPU {
        utils::cudaExecuteAndTimeFunction([&]{
            matMulCUDA(
                A.data(), B.data(), C_gpu.data(),
                Constants::n, Constants::m, Constants::k, tileWidth
            );
        })
    };
    std::cout << "GPU version elapsed time: " << secondsGPU << "seconds\n";

    // Check if results are the same
    bool ok { utils::almostEqual(
        C_cpu, C_gpu, Constants::n, Constants::k, 1e-6f, 1e-6f)
    };
    std::cout << (ok ? "OK\n" : "MISMATCH!\n");

    return 0;
}
