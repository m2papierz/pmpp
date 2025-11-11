#include "utils.hpp"
#include "kernels.cuh"
#include "functions.cuh"

#include <iostream>

namespace Constants {
    constexpr int height { 4096 };
    constexpr int width { 4096 };
    constexpr int filterDim { 2*FILTER_RADIUS + 1};
}

int main() {
    // Initiate input and filter
    std::vector<float> inArray(Constants::height * Constants::width);
    std::vector<float> filter(Constants::filterDim * Constants::filterDim);
    for (auto& x : inArray) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& w : filter) w = static_cast<float>(rand()) / RAND_MAX;

    // CPU reference
    std::vector<float> outCPU(Constants::height * Constants::width);;
    double secondsCPU {
        utils::executeAndTimeFunction([&]{
            conv2dCPU(
                inArray.data(), filter.data(), outCPU.data(),
                FILTER_RADIUS, Constants::height, Constants::width
            );
        })
    };
    std::cout << "CPU version elapsed time: " << secondsCPU << "seconds\n";

    auto runAndCheck = [&](auto gpuFunc, const char* name) {
        std::vector<float> outGPU(Constants::height * Constants::width);

        // Pass copies as kernels modify in-place
        auto inCopy { inArray };
        auto filterCopy { filter };

        float seconds {
            utils::cudaExecuteAndTimeFunction([&]{
                gpuFunc(
                    inCopy.data(), filterCopy.data(), outGPU.data(),
                    FILTER_RADIUS, Constants::height, Constants::width
                );
            })
        };
        std::cout << '\n' << name << " elapsed time: " << seconds << " seconds\n";
        const bool ok = utils::matricesAlmostEqual(
            outCPU, outGPU, Constants::height, Constants::width, 1e-6f, 1e-6f
        );
        std::cout << (ok ? "OK" : "MISMATCH!") << "\n";
    };

    runAndCheck(conv2d, "Basic GPU");
    runAndCheck(conv2dConstMem, "Constant memory GPU");
    runAndCheck(conv2dTiledIn, "Tiled IN GPU");
    runAndCheck(conv2dTiledOut, "Tiled OUT GPU");

    return 0;
}
