#include "utils.hpp"
#include "reductions.cuh"

namespace {
    constexpr int lengthSimple { 2048 };
    constexpr int length { 10000000 };
}

template <typename GpuFunc>
auto runAndCheck(
    GpuFunc gpuFunc,
    const char* name,
    const std::vector<float>& inputData,
    int length,
    float outCPU
) {
    float outGPU { 0.0f };

    // Pass copies as kernels modify in-place
    auto inCopy { inputData };
    float parallelTime {
        utils::cudaExecuteAndTimeFunction([&]{
            gpuFunc(inCopy.data(), &outGPU, length);
        }, 0, 1)
    };
    std::cout << name << " elapsed time: " << parallelTime << "s\n";
    if (!utils::almostEqual(outCPU, outGPU, 1e-3, 1e-3)) {
        std::cerr << "Mismatch with reference in " << name << "!\n"; std::exit(1);
    }
};

void benchmarkSumSimple() {
    float outCPU { 0.0f };
    std::vector<float> inputData(lengthSimple);
    for (auto& x : inputData) x = static_cast<float>(rand()) / RAND_MAX;

    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            reductionSumSerial(inputData.data(), &outCPU, lengthSimple);
        }, 0, 1)
    };
    std::cout << "\nCPU with " << lengthSimple << " elements elapsed time: " << sequentialTime << "s\n";

    runAndCheck(reductionSimple, "Naive Kernel", inputData, lengthSimple, outCPU);
    runAndCheck(reductionConvergent, "Convergent Kernel", inputData, lengthSimple, outCPU);
    runAndCheck(reductionConvergentSharedMem, "Convergent Kernel with shared memory", inputData, lengthSimple, outCPU);
}

void benchmarkSumSegmented() {
    float outCPU { 0.0f };
    static std::vector<float> inputData(length);
    for (auto& x : inputData) x = static_cast<float>(rand()) / RAND_MAX;

    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            reductionSumSerial(inputData.data(), &outCPU, length);
        }, 0, 1)
    };
    std::cout << "\nCPU with " << length << " elements elapsed time: " << sequentialTime << "s\n";

    runAndCheck(reductionSegmented, "Segmented Kernel", inputData, length, outCPU);
    runAndCheck(reductionCoarsed, "Segmented Coarsed Kernel", inputData, length, outCPU);
}

void benchmarkMaxSegmented() {
    float outCPU { 0.0f };
    static std::vector<float> inputData(length);
    for (auto& x : inputData) x = static_cast<float>(rand()) / RAND_MAX;

    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            reductionMaxSerial(inputData.data(), &outCPU, length);
        }, 0, 1)
    };
    std::cout << "\nCPU max reductions with " << length << " elements elapsed time: " << sequentialTime << "s\n";

    runAndCheck(reductionMaxCoarsed, "Segmented MAX Kernel", inputData, length, outCPU);
}

int main() {
    benchmarkSumSimple();
    benchmarkSumSegmented();
    benchmarkMaxSegmented();

    return 0;
}
