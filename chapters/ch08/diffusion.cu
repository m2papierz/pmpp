#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE_STEP 8
#define OUT_TILE_DIM_LAP 30
#define IN_TILE_DIM_LAP (OUT_TILE_DIM_LAP + 2)

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                \
                        << " (" << __FILE__ << ":" << __LINE__ << ")\n";            \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

namespace utils {
    inline unsigned int cdiv(const int x, const int div) {
        return (x + div - 1) / div;
    }
}

__global__ void laplacianKernel3D(
    const float* __restrict__ u,
    float* __restrict__ lap,
    int n,
    float inv_h2
) {
    int depStart { static_cast<int>(blockIdx.z*OUT_TILE_DIM_LAP) };
    int row { static_cast<int>(blockIdx.y*OUT_TILE_DIM_LAP + threadIdx.y - 1) };
    int col { static_cast<int>(blockIdx.x*OUT_TILE_DIM_LAP + threadIdx.x - 1) };
    float inPrev {};

    // Collaboratively load data into shared memory
    __shared__ float inCurr_s[IN_TILE_DIM_LAP][IN_TILE_DIM_LAP];
    float inCurr {};
    float inNext {};

    if (
        depStart - 1 >= 0 && depStart - 1 < n &&
        row >= 0 && row < n &&
        col >= 0 && col < n
    ) {
        inPrev = u[(depStart - 1)*n*n + row*n + col];
    }
    if (
        depStart >= 0 && depStart < n &&
        row >= 0 && row < n &&
        col >= 0 && col < n
    ) {
        inCurr = u[depStart*n*n + row*n + col];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for (int dep { depStart }; dep < depStart + OUT_TILE_DIM_LAP; ++dep) {
        if(
            dep + 1 >= 0 && dep + 1 < n &&
            row >= 0 && row < n &&
            col >= 0 && col < n
        ) {
            inNext = u[(dep + 1)*n*n + row*n + col];
        }
        __syncthreads();

        if (
            dep >= 1 && dep < n - 1 &&
            row >= 1 && row < n - 1 &&
            col >= 1 && col < n - 1
        ) {
            if(
                threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM_LAP - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM_LAP - 1
            ) {
                lap[dep*n*n + row*n + col] = inv_h2 * (
                    inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    inCurr_s[threadIdx.y][threadIdx.x + 1] +
                    inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    inCurr_s[threadIdx.y + 1][threadIdx.x] +
                    inPrev + inNext - 6.0f * inCurr
                );                  
            }
        }
        __syncthreads();

        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

__global__ void allenCahnStepKerne3D(
    const float* __restrict__ u,
    const float* __restrict__ lap,
    float* __restrict__ u_next,
    int dim,
    float dt,
    float eps
) {
    int dep { static_cast<int>(blockIdx.z*blockDim.z + threadIdx.z) };
    int row { static_cast<int>(blockIdx.y*blockDim.y + threadIdx.y) };
    int col { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };

    // First: kill threads that are outside the domain
    if (dep >= dim || row >= dim || col >= dim) {
        return;
    }

    // Set the global index
    int idx { dep*dim*dim + row*dim + col};

    // Boundary: keep value fixed (Dirchlet)
    if (
        dep == 0 || dep == dim - 1 ||
        row == 0 || row == dim - 1 ||
        col == 0 || col == dim - 1
    ) {
        u_next[idx] = u[idx];
    } else {
        u_next[idx] = u[idx] + dt*(
            eps*eps* lap[idx] + u[idx] - u[idx]*u[idx]*u[idx]
        );
    }
};

extern "C"
void allenCahnStep(
    float* __restrict__ u,
    int dim,
    float dh,
    float dt,
    float eps
) {
    // Allocate memory on the device
    std::size_t tensorSize { static_cast<std::size_t>(dim*dim*dim*sizeof(float)) };

    float *d_u { nullptr }, *d_lap { nullptr }, *d_u_next {nullptr};
    CUDA_CHECK(cudaMalloc(&d_u, tensorSize));
    CUDA_CHECK(cudaMalloc(&d_lap, tensorSize));
    CUDA_CHECK(cudaMalloc(&d_u_next, tensorSize));

    // Move data from host to device
    CUDA_CHECK(cudaMemcpy(d_u, u, tensorSize, cudaMemcpyHostToDevice));

    // Call kernels
    dim3 blockSizeLap(IN_TILE_DIM_LAP, IN_TILE_DIM_LAP, 1);
    dim3 gridSizeLap(
        utils::cdiv(dim, OUT_TILE_DIM_LAP),
        utils::cdiv(dim, OUT_TILE_DIM_LAP),
        utils::cdiv(dim, OUT_TILE_DIM_LAP)
    );

    dim3 blockSizeStep(BLOCK_SIZE_STEP, BLOCK_SIZE_STEP, BLOCK_SIZE_STEP);
    dim3 gridSizeStep(
        utils::cdiv(dim, BLOCK_SIZE_STEP),
        utils::cdiv(dim, BLOCK_SIZE_STEP),
        utils::cdiv(dim, BLOCK_SIZE_STEP)
    );

    float inv_h2 { 1.0f / (dh*dh) };
    laplacianKernel3D<<<gridSizeLap, blockSizeLap>>>(d_u, d_lap, dim, inv_h2);
    CUDA_CHECK(cudaGetLastError());

    allenCahnStepKerne3D<<<gridSizeStep, blockSizeStep>>>(d_u, d_lap, d_u_next, dim, dt, eps);
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_u, d_u_next);

    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(u, d_u, tensorSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_lap));
    CUDA_CHECK(cudaFree(d_u_next));
}

extern "C"
void cleanupCuda() {
    // Best-effort device cleanup
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "CUDA cleanup error: " << cudaGetErrorString(err)
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n";
    }
}
