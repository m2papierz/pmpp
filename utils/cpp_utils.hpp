#pragma once
#include <algorithm>  // std::min
#include <chrono>     // std::chrono::high_resolution_clock
#include <cmath>      // std::fabs
#include <cstddef>    // std::size_t
#include <span>       // std::span
#include <vector>     // std::vector
#include <functional>


namespace utils {
    inline unsigned int cdiv(const int x, const int div) {
        return (x + div - 1) / div;
    }

    template <typename F>
    double executeAndTimeFunction(F&& func) {
        auto start { std::chrono::high_resolution_clock::now() };
        func();
        auto end { std::chrono::high_resolution_clock::now() };
        std::chrono::duration<double> diff { end - start };
        return diff.count();
    }

    inline void matMulCPU(
        const float* A,
        const float* B,
        float* C,
        int n, int m, int k
    ) {
        for (int row { 0 }; row < n; ++row) {
            for (int col { 0 }; col < k; ++col) {
                float pValue { 0 };
                for (int i { 0 }; i < m; ++i) {
                    pValue += A[row * m + i] * B[i * k + col];
                }
                C[row * k + col] = pValue;
            }
        }
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

    namespace detail {
        template <typename T>
        inline bool almostEqualSpan(
            std::span<const T> A,
            std::span<const T> B,
            T atol = T(1e-6),
            T rtol = T(1e-6)
        ) {
            static_assert(
                std::is_floating_point_v<T>,
                "almostEqualSpan requires a floaring-point type"
            );

            if (A.size() != B.size()) {
                return false;
            }

            const std::size_t n { A.size() };
            for (std::size_t i { 0 }; i < n; ++i) {
                T diff { std::fabs(A[i] - B[i]) };
                T tol { atol + rtol * std::fabs(B[i]) };
                if (diff > tol) {
                    return false;
                }
            }
            return true;
        }
    } // namespace detail

    // 1D vectors
    inline bool almostEqual(
        const std::vector<float>& A,
        const std::vector<float>& B,
        float atol = 1e-6f,
        float rtol = 1e-6f
    ) {
        return detail::almostEqualSpan<float>(
            std::span<const float>(A),
            std::span<const float>(B),
            atol,
            rtol
        );
    }

    // 2D vectors
    inline bool almostEqual(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::size_t rows,
        std::size_t cols,
        float atol = 1e-6f,
        float rtol = 1e-6f
    ) {
        const std::size_t n { rows * cols };
        if (A.size() != B.size() || A.size() != n) {
            return false;
        }

        return detail::almostEqualSpan<float>(
            std::span<const float>(A),
            std::span<const float>(B),
            atol,
            rtol
        );
    }

    // 3D vectors
    inline bool almostEqual(
        const std::vector<float>& A,
        const std::vector<float>& B,
        std::size_t depth,
        std::size_t height,
        std::size_t width,
        float atol = 1e-6f,
        float rtol = 1e-6f
    ) {
        const std::size_t n { depth * height * width };
        if (A.size() != B.size() || A.size() != n) {
            return false;
        }

        return detail::almostEqualSpan<float>(
            std::span<const float>(A),
            std::span<const float>(B),
            atol,
            rtol
        );
    }
} // namespace utils
