#include "merge.cuh"
#include "utils.hpp"

namespace {
    constexpr int m { 32768 };
    constexpr int n { 32768 };
    constexpr int benchWarmupIters { 0 };
    constexpr int benchRepeatIters { 1 };
}

template <typename GpuFunc>
auto runAndCheck(
    GpuFunc gpuFunc,
    const char* name,
    const std::vector<int>& A,
    const std::vector<int>& B,
    const std::vector<int>& outCPU
) {
    std::vector<int> outGPU(m + n);

    // Pass copies as kernels modify in-place
    auto ACpy { A };
    auto BCpy { B };
    float parallelTime {
        utils::cudaExecuteAndTimeFunction([&]{
            gpuFunc(ACpy.data(), m, BCpy.data(), n, outGPU.data());
        }, benchWarmupIters, benchRepeatIters)
    };
    std::cout << name << " elapsed time: " << parallelTime << "s\n";
    if (!utils::almostEqual(outCPU, outGPU, 1e-3, 1e-3)) {
        std::cerr << "Mismatch with reference in " << name << "!\n"; std::exit(1);
    }
};

int main() {
    std::vector<int> A(m);
    std::vector<int> B(n);
    std::vector<int> outCPU(m + n);
    for (auto& x : A) x = static_cast<int>(rand()) / RAND_MAX;
    for (auto& x : B) x = static_cast<int>(rand()) / RAND_MAX;

    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            mergeSequential(A.data(), m, B.data(), n, outCPU.data());
        }, benchWarmupIters, benchRepeatIters)
    };
    std::cout << "\nCPU elapsed time: " << sequentialTime << "s\n";

    runAndCheck(mergeBasic, "Naive Kernel", A, B, outCPU);

    return 0;
}
