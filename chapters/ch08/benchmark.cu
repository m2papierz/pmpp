#include "utils.hpp"
#include "stencil_sweep.cuh"
#include <vector>
#include <iostream>

namespace {
    constexpr int tensorDim { 512 };
    constexpr int c0 { 1 };
    constexpr int c1 { 1 };
    constexpr int c2 { 1 };
    constexpr int c3 { 1 };
    constexpr int c4 { 1 };
    constexpr int c5 { 1 };
    constexpr int c6 { 1 };
}

int main() {
    // Generate input tensor
    std::vector<float> inTensor(tensorDim*tensorDim*tensorDim);
    for (auto& x : inTensor) x = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference
    std::vector<float> outCPU(tensorDim*tensorDim*tensorDim);
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            stencilCPU(
                inTensor.data(), outCPU.data(), tensorDim,
                c0, c1, c2, c3, c4, c5, c6
            );
        })
    };
    std::cout << "CPU version elapsed time: " << secondsCPU << "seconds\n";

    auto runAndCheck = [&](auto gpuFunc, const char* name) {
        std::vector<float> outGPU(tensorDim*tensorDim*tensorDim);

        // Pass copy as kernels modify in-place
        auto inCopy { inTensor };

        float seconds {
            utils::cudaExecuteAndTimeFunction([&]{
                gpuFunc(
                    inCopy.data(), outGPU.data(), tensorDim,
                    c0, c1, c2, c3, c4, c5, c6
                );
            })
        };
        std::cout << name << " elapsed time: " << seconds << " seconds\n";
        if (!utils::almostEqual(outCPU, outGPU, tensorDim, tensorDim, tensorDim, 1e-6f, 1e-6f)) {
            std::cerr << "Mismatch with reference!\n"; std::exit(1);
        }
    };

    runAndCheck(stencilNaive, "Naive GPU");
    runAndCheck(stencilSharedMem, "Shared memory memory GPU");
    runAndCheck(stencilThreadCoarsening, "Thread coarsening GPU");
    runAndCheck(stencilRegisterTiling, "Register tiling GPU");

    return 0;
}
