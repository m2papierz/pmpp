#include "histogram.cuh"
#include "utils.hpp"

namespace {
    constexpr int length { 100000000 };
    constexpr int binSize { 4 };
    constexpr int numBins { (26 + binSize - 1) / binSize };
};

int main() {
    std::vector<char> inputData(length);
    for (int i { 0 }; i < length; ++i){
        inputData[i] = static_cast<char>(Random::get(0, 26));
    }

    std::vector<unsigned int> outCPU(numBins);
    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            histogramSequential(inputData.data(), length, outCPU.data());
        })
    };
    std::cout << "\nCPU version elapsed time: " << sequentialTime << "seconds\n";

    auto runAndCheck = [&](auto gpuFunc, const char* name) {
        std::vector<unsigned int> outGPU(numBins);

        // Pass copies as kernels modify in-place
        auto inCopy { inputData };
        float parallelTime {
            utils::cudaExecuteAndTimeFunction([&]{
                gpuFunc(inCopy.data(), length, outGPU.data());
            })
        };
        std::cout << name << " elapsed time: " << parallelTime << " seconds\n";
        if (!utils::almostEqual(outCPU, outGPU, 1e-6, 1e-6)) {
            std::cerr << "Mismatch with reference!\n"; std::exit(1);
        }
    };

    runAndCheck(histogramNaive, "Kernel Naive");
    runAndCheck(histogramPrivate, "Kernel Private");
    runAndCheck(histogramSharedMem, "Kernel Private");

    return 0;
}
