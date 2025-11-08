#pragma once
#include <chrono>
#include <functional>

namespace utils {
    template <typename F>
    double executeAndTimeFunction(F&& func) {
        auto start { std::chrono::high_resolution_clock::now() };
        func();
        auto end { std::chrono::high_resolution_clock::now() };
        std::chrono::duration<double> diff { end - start };
        return diff.count();
    }

    inline bool matricesAlmostEqual(
        const float* A,
        const float* B,
        std::size_t rows,
        std::size_t cols,
        float atol = 1e-6f,
        float rtol = 1e-6f
    ) {
        const std::size_t n = rows * cols;
        for (std::size_t i = 0; i < n; ++i) {
            float diff = std::fabs(A[i] - B[i]);
            float tol = atol + rtol * std::fabs(B[i]);
            if (diff > tol) { return false; }
        }
        return true;
    }

    // convenience for std::vector<float>
    inline bool matricesAlmostEqual(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::size_t rows,
        std::size_t cols,
        float atol = 1e-6f,
        float rtol = 1e-6f
    ) {
        if (A.size() != B.size() || A.size() != rows * cols) {
            return false;
        }
        return matricesAlmostEqual(A.data(), B.data(), rows, cols, atol, rtol);
    }

    inline void transposeMatrixTiled(
        const std::vector<float>& inputMat,
        std::vector<float>& outputMat,
        int width,
        int height,
        int tileSize = 256
    ) {
        for (int i_tile { 0 }; i_tile < width; i_tile += tileSize) {
            const int i_max { std::min(i_tile + tileSize, width) };
            for (int j_tile { 0 }; j_tile < height; j_tile += tileSize) {
                const int j_max { std::min(j_tile + tileSize, height) };

                for (int i { i_tile }; i < i_max; ++i) {
                    const int in_base { i * height };
                    for (int j {j_tile}; j < j_max; ++j) {
                        outputMat[j * width + i] = inputMat[in_base + j];
                    }
                }

            }
        }
    }
} // namespace utils
