#include "scan.cuh"
#include "utils.hpp"

namespace {
    constexpr int lengthSimple { 1024 };
    constexpr int lengthHierarchical { 1048576 };
    constexpr int benchWarmupIters { 5 };
    constexpr int benchRepeatIters { 5 };
}

template <typename GpuFunc>
auto runAndCheck(
    GpuFunc gpuFunc,
    const char* name,
    const std::vector<float>& inputData,
    int length,
    const std::vector<float>& outCPU
) {
    std::vector<float> outGPU(length);

    // Pass copies as kernels modify in-place
    auto inCopy { inputData };
    float parallelTime {
        utils::cudaExecuteAndTimeFunction([&]{
            gpuFunc(inCopy.data(), outGPU.data(), length);
        }, benchWarmupIters, benchRepeatIters)
    };
    std::cout << name << " elapsed time: " << parallelTime << "s\n";
    if (!utils::almostEqual(outCPU, outGPU, 1e-3, 1e-3)) {
        std::cerr << "Mismatch with reference in " << name << "!\n"; std::exit(1);
    }
};

void benchmarkSimple() {
    std::vector<float> outCPU(lengthSimple);
    std::vector<float> inputData(lengthSimple);
    for (auto& x : inputData) x = static_cast<float>(rand()) / RAND_MAX;

    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            scanSequential(inputData.data(), outCPU.data(), lengthSimple);
        }, benchWarmupIters, benchRepeatIters)
    };
    std::cout << "\nCPU with " << lengthSimple << " elements elapsed time: " << sequentialTime << "s\n";

    runAndCheck(koggeStone, "Kogge-Stone Kernel", inputData, lengthSimple, outCPU);
    runAndCheck(koggeStoneDoubleBuffer, "Kogge-Stone 2-buffers Kernel", inputData, lengthSimple, outCPU);
    runAndCheck(brentKung, "Brent-Kung Kernel", inputData, lengthSimple, outCPU);
    runAndCheck(coarsenedThreePhase, "Coarsened Three-phase Kernel", inputData, lengthSimple, outCPU);
    runAndCheck(thrustScan, "Thrust Scan", inputData, lengthSimple, outCPU);
}

void benchmarkHierarchical() {
    std::vector<float> outCPU(lengthHierarchical);
    std::vector<float> inputData(lengthHierarchical);
    for (auto& x : inputData) x = static_cast<float>(rand()) / RAND_MAX;

    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            scanSequential(inputData.data(), outCPU.data(), lengthHierarchical);
        }, benchWarmupIters, benchRepeatIters)
    };
    std::cout << "\nCPU with " << lengthHierarchical << " elements elapsed time: " << sequentialTime << "s\n";

    runAndCheck(hierarchicalScan, "Hierarchical Kernel", inputData, lengthHierarchical, outCPU);
    runAndCheck(hierarchicalDominoScan, "Hierarchical Domino Kernel", inputData, lengthHierarchical, outCPU);
    runAndCheck(thrustScan, "Thrust Scan", inputData, lengthHierarchical, outCPU);
}


int main() {
    benchmarkSimple();
    benchmarkHierarchical();

    return 0;
}
