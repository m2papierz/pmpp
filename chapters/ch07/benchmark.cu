#include "utils.hpp"
#include "kernels/kernels2d.cuh"
#include "kernels/kernels3d.cuh"
#include "functions.cuh"

#include <cassert>
#include <cstdlib>
#include <vector>
#include <iostream>

namespace Constants2D {
    constexpr int height { 4096 };
    constexpr int width { 4096 };
    constexpr int filterDim { 2*config2d::FILTER_RADIUS + 1};
}

namespace Constants3D {
    constexpr int depth { 256 };
    constexpr int height { 256 };
    constexpr int width { 256 };
    constexpr int filterDim { 2*config3d::FILTER_RADIUS + 1 };
}

void run2DKernels() {
    using namespace Constants2D;

    // Initiate input and filter
    std::vector<float> inArray(height * width);
    std::vector<float> filter(filterDim * filterDim);
    for (auto& x : inArray) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& w : filter) w = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference
    std::vector<float> outCPU(height * width);
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            conv2dCPU(
                inArray.data(), filter.data(), outCPU.data(),
                config2d::FILTER_RADIUS, height, width
            );
        })
    };
    std::cout << "\n2D CPU version elapsed time: " << secondsCPU << "seconds\n";

    auto runAndCheck = [&](auto gpuFunc, const char* name) {
        std::vector<float> outGPU(height * width);

        // Pass copies as kernels modify in-place
        auto inCopy { inArray };
        auto filterCopy { filter };

        float seconds {
            utils::cudaExecuteAndTimeFunction([&]{
                gpuFunc(
                    inCopy.data(), filterCopy.data(), outGPU.data(),
                    config2d::FILTER_RADIUS, height, width
                );
            })
        };
        std::cout << name << " elapsed time: " << seconds << " seconds\n";
        if (!utils::almostEqual(outCPU, outGPU, height, width, 1e-6f, 1e-6f)) {
            std::cerr << "Mismatch with reference!\n"; std::exit(1);
        }
    };

    runAndCheck(conv2d, "Basic 2D GPU");
    runAndCheck(conv2dConstMem, "Constant 2D memory GPU");
    runAndCheck(conv2dTiledIn, "Tiled 2D IN GPU");
    runAndCheck(conv2dTiledOut, "Tiled 2D OUT GPU");
    runAndCheck(conv2dTiledCached, "Tiled 2D CACHED GPU");
}

void run3DKernels() {
    using namespace Constants3D;

    const std::size_t N { static_cast<std::size_t>(depth * height * width) };

    // Initialize input and filter
    std::vector<float> inArray(N);
    std::vector<float> filter(filterDim * filterDim * filterDim);

    for (auto& x : inArray) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& w : filter)  w = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference (timing only)
    std::vector<float> outCPU(N);
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            conv3dCPU(
                inArray.data(), filter.data(), outCPU.data(),
                config3d::FILTER_RADIUS,
                height, width, depth
            );
        })
    };
    std::cout << "\n3D CPU version elapsed time: " << secondsCPU << " seconds\n";

    auto runAndCheck = [&](auto gpuFunc, const char* name) {
        std::vector<float> outGPU(N);

        // Pass copies as kernels may modify in-place
        auto inCopy = inArray;
        auto filterCopy = filter;

        float seconds {
            utils::cudaExecuteAndTimeFunction([&]{
                gpuFunc(
                    inCopy.data(), filterCopy.data(), outGPU.data(),
                    config3d::FILTER_RADIUS,
                    height, width, depth
                );
            })
        };
        std::cout << name << " elapsed time: " << seconds << " seconds\n";
        if (!utils::almostEqual(outCPU, outGPU, height, width, depth, 1e-6f, 1e-6f)) {
            std::cerr << "Mismatch with reference!\n"; std::exit(1);
        }
    };

    runAndCheck(conv3d, "Basic 3D GPU");
    runAndCheck(conv3dConstMem, "Constant memory 3D GPU");
    runAndCheck(conv3dTiled, "Tiled 3D GPU");
}

int main() {
    run2DKernels();

    run3DKernels();

    return 0;
}
