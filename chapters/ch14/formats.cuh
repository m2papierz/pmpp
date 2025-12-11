#pragma once

#include <vector>
#include <unordered_set>
#include <cstddef>
#include <cuda_runtime.h>

#include "utils.hpp"

struct COOMatrix {
    int rows{};
    int cols{};
    int numNonzeros{};

    std::vector<int> rowIdx;
    std::vector<int> colIdx;
    std::vector<float> values;

    COOMatrix() = default;

    // Construct a random sparse matrix with given density (default 2%)
    explicit COOMatrix(
        int r,
        int c,
        double density = 0.02,
        float minVal = 0.01f,
        float maxVal = 1.0f
    )
        : rows{r}, cols{c}, numNonzeros{0}
    {
        const std::size_t total {
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(cols)
        };
        const std::size_t nnz { static_cast<std::size_t>(total*density) };

        rowIdx.reserve(nnz);
        colIdx.reserve(nnz);
        values.reserve(nnz);

        struct PairHash {
            std::size_t operator()(const std::pair<int, int>& p) const noexcept {
                return (static_cast<std::size_t>(p.first) * 73856093u) ^
                       (static_cast<std::size_t>(p.second) * 19349663u);
            }
        };

        std::unordered_set<std::pair<int, int>, PairHash> used;
        used.reserve(nnz * 2);

        while (used.size() < nnz) {
            int rIdx = Random::get<int>(0, rows - 1);
            int cIdx = Random::get<int>(0, cols - 1);

            if (used.emplace(rIdx, cIdx).second) { // inserted new coord
                rowIdx.push_back(rIdx);
                colIdx.push_back(cIdx);
                values.push_back(Random::get<float>(minVal, maxVal));
                ++numNonzeros;
            }
        }
    }
};

struct DeviceCOOMatrix {
    int numNonzeros{};
    const int* rowIdx{};
    const int* colIdx{};
    const float* values{};
};

struct COOMatrixDevice {
    DeviceCOOMatrix dev{};
    int* d_rowIdx{nullptr};
    int* d_colIdx{nullptr};
    float* d_values{nullptr};

    COOMatrixDevice() = default;
    explicit COOMatrixDevice(const COOMatrix& h);
    ~COOMatrixDevice();

    COOMatrixDevice(const COOMatrixDevice&) = delete;
    COOMatrixDevice& operator=(const COOMatrixDevice&) = delete;
    COOMatrixDevice(COOMatrixDevice&& other) noexcept;
    COOMatrixDevice& operator=(COOMatrixDevice&& other) noexcept;
};
