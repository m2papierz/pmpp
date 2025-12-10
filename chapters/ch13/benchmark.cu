#include "sorting.cuh"
#include "utils.hpp"

namespace {
    constexpr int n { 1000000 };
    constexpr int benchWarmupIters { 0 };
    constexpr int benchRepeatIters { 1 };
}

template <typename GpuFunc>
auto runAndCheck(
    GpuFunc gpuFunc,
    const char* name,
    const std::vector<unsigned int>& inputArr,
    const std::vector<unsigned int>& outCPU
) {
    std::vector<unsigned int> outGPU(n);

    // Pass copies as kernels modify in-place
    auto inCopy { inputArr };
    float parallelTime {
        utils::cudaExecuteAndTimeFunction([&]{
            gpuFunc(inCopy.data(), outGPU.data(), n);
        }, benchWarmupIters, benchRepeatIters)
    };
    std::cout << name << " elapsed time: " << parallelTime << "s\n";
    if (outCPU != outGPU) {
        std::cerr << "Mismatch with reference in " << name << "!\n"; std::exit(1);
    }
};

int main() {
    std::vector<unsigned int> inputArr(n);
    std::vector<unsigned int> outCPU(n);
    // Initialize with full 32-bit unsigned range
    for (auto& x : inputArr) {
        x = Random::get<unsigned int>(
            0u, std::numeric_limits<unsigned int>::max() - 1
        );
    }

    double sequentialTime {
        utils::executeAndTimeFunction([&]{
            radixSortCPU(inputArr.data(), outCPU.data(), n);
        }, benchWarmupIters, benchRepeatIters)
    };
    std::cout << "\nCPU elapsed time: " << sequentialTime << "s\n";

    if (!std::is_sorted(outCPU.begin(), outCPU.end())) {
        std::cerr << "CPU result is not sorted!\n"; 
        return 1;
    }

    runAndCheck(radixSort, "Basic Kernel", inputArr, outCPU);
    runAndCheck(radixSortCoalesced, "Coalesced Kernel", inputArr, outCPU);
    runAndCheck(radixSortCoalescedMultibit, "Coalesced Multibit Kernel", inputArr, outCPU);
    runAndCheck(radixSortCoalescedCoarse, "Thread Coarsed Kernel", inputArr, outCPU);
    runAndCheck(mergeSort, "Merge Kernel", inputArr, outCPU);

    return 0;
}
