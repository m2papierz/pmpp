#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <iostream>


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
    template <typename F>
    float cudaExecuteAndTimeFunction(
        F&& func,
        int warmupIters = 5,
        int repeatIters = 5
    ) {
        // Warmup
        for (int i { 0 }; i < warmupIters; ++i) {
            func();
        }
        CUDA_CHECK(cudaDeviceSynchronize()); // ensure warmup work is done

        cudaEvent_t start {}, stop {};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start, 0));
            for (int i { 0 }; i < repeatIters; ++i) {
            func();
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float diff { 0.0 };
        CUDA_CHECK(cudaEventElapsedTime(&diff, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Return average time per call in seconds
        return (diff * 1e-3f) / repeatIters;
    }
} // namespace utils
