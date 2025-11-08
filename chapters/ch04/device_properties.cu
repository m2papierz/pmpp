#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    int deviceCount { 0 };
    cudaError_t error_id { cudaGetDeviceCount(&deviceCount) };

    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount error: " << cudaGetErrorString(error_id) << '\n';
        exit(EXIT_FAILURE);
    }

    std::cout << "Detected " << deviceCount << " CUDA capable device(s)" << '\n';

    cudaDeviceProp deviceProp {};
    cudaGetDeviceProperties(&deviceProp, 0);

    std::cout << "\nDevice: " << deviceProp.name << '\n';
    std::cout << "  Total amount of global memory:     " << std::fixed << std::setprecision(2)
                << static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024) << " GB\n";
    std::cout << "  Number of SMs:                     " << deviceProp.multiProcessorCount << '\n';
    std::cout << "  Total amount of constant memory:   " << deviceProp.totalConstMem << " bytes\n";
    std::cout << "  Total shared memory per block:     " << deviceProp.sharedMemPerBlock << " bytes\n";
    std::cout << "  Total registers per block:         " << deviceProp.regsPerBlock << '\n';
    std::cout << "  Warp size:                         " << deviceProp.warpSize << '\n';
    std::cout << "  Max threads per block:             " << deviceProp.maxThreadsPerBlock << '\n';
    std::cout << "  Max threads per SM:                " << deviceProp.maxThreadsPerMultiProcessor << '\n';
    std::cout << "  Max dimensions of a block:         " << deviceProp.maxThreadsDim[0] << " x "
        << deviceProp.maxThreadsDim[1] << " x "
        << deviceProp.maxThreadsDim[2] << '\n';
    std::cout << "  Max dimensions of a grid:          " << deviceProp.maxGridSize[0] << " x "
        << deviceProp.maxGridSize[1] << " x "
        << deviceProp.maxGridSize[2] << '\n';
    std::cout << "  Clock rate:                        " << std::fixed << std::setprecision(2)
        << deviceProp.clockRate * 1e-6f << " GHz\n";
    std::cout << "  Memory clock rate:                 " << deviceProp.memoryClockRate / 1000 << "MHz\n";
    std::cout << "  Memory bus width:                  " << deviceProp.memoryBusWidth << "-bit\n";
    std::cout << "  L2 cache size:                     " << deviceProp.l2CacheSize << " bytes\n";

    return 0;
}
