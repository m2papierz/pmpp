#include "sparse.cuh"

namespace {
    constexpr int rows { 10000 };
    constexpr int cols { 10000 };
    constexpr double density { 0.01 };

    constexpr int benchWarmupIters { 5 };
    constexpr int benchRepeatIters { 5 };
}


void runAndCheckCOO() {
    COOMatrix A(rows, cols, density);
    std::vector<float> x(cols);
    std::vector<float> yGPU(A.rows);
    std::vector<float> yCPU(A.rows);

    for (auto& v : x) { v = Random::get<float>(0.0f, 1.0f); }
    
    double sequentialTime = utils::executeAndTimeFunction([&]{
        spmv_coo_cpu(A, x.data(), yCPU.data());
    }, benchWarmupIters, benchRepeatIters);
    std::cout << "\nCPU (COO) elapsed time: " << sequentialTime << "s\n";

    float parallelTime = utils::cudaExecuteAndTimeFunction([&]{
        spmv_coo(A, x.data(), yGPU.data());
    }, benchWarmupIters, benchRepeatIters);

    std::cout << "GPU SpMV (COO) elapsed time: " << parallelTime << "s\n";

    // compare results
    const float eps = 1e-3f;
    for (int i = 0; i < A.rows; ++i) {
        if (std::fabs(yCPU[i] - yGPU[i]) > eps) {
            std::cerr << "Mismatch! " << '\n';
            std::exit(1);
        }
    }
}

void runAndCheckCRS() {
    CRSMatrix A(rows, cols, density);
    std::vector<float> x(cols);
    std::vector<float> yGPU(rows);
    std::vector<float> yCPU(rows);

    for (auto& v : x) { v = Random::get<float>(0.0f, 1.0f); }
    
    double sequentialTime = utils::executeAndTimeFunction([&]{
        spmv_crs_cpu(A, x.data(), yCPU.data());
    }, benchWarmupIters, benchRepeatIters);
    std::cout << "\nCPU (CRS) elapsed time: " << sequentialTime << "s\n";

    float parallelTime = utils::cudaExecuteAndTimeFunction([&]{
        spmv_crs(A, x.data(), yGPU.data());
    }, benchWarmupIters, benchRepeatIters);

    std::cout << "GPU SpMV (CRS) elapsed time: " << parallelTime << "s\n";

    // compare results
    const float eps = 1e-3f;
    for (int i = 0; i < rows; ++i) {
        if (std::fabs(yCPU[i] - yGPU[i]) > eps) {
            std::cerr << "Mismatch! " << '\n';
            std::exit(1);
        }
    }
}

int main() {
    runAndCheckCOO();
    runAndCheckCRS();

    return 0;
}
