#pragma once

#include <vector>
#include <unordered_set>
#include <cstddef>
#include <cuda_runtime.h>

#include "utils.hpp"

// ---------- COO format ----------

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

// ---------- CRS format ----------

struct CRSMatrix {
    int numRows{};
    int numCols{};
    std::size_t numNonzeros{};

    std::vector<int> rowPtrs;
    std::vector<int> colIdx;
    std::vector<float> values;

    CRSMatrix() = default;

    explicit CRSMatrix(
        int rows,
        int cols,
        double density = 0.02,
        float minVal = 0.01f,
        float maxVal = 1.0f
    )
        : numRows{rows}, numCols{cols}, numNonzeros{0}
    {
        const std::size_t total {
            static_cast<std::size_t>(numRows) *
            static_cast<std::size_t>(numCols)
        };
        const std::size_t nnzTarget { static_cast<std::size_t>(total*density) };

        std::vector<int> tmpRowIdx;
        std::vector<int> tmpColIdx;
        std::vector<float> tmpValues;

        tmpRowIdx.reserve(nnzTarget);
        tmpColIdx.reserve(nnzTarget);
        tmpValues.reserve(nnzTarget);

        // Pack (row,col) into uint64 to detect duplicates
        auto pack = [](int r, int c) -> std::uint64_t {
            return (static_cast<std::uint64_t>(r) << 32) |
                   static_cast<std::uint64_t>(static_cast<std::uint32_t>(c));
        };

        std::unordered_set<std::uint64_t> used;
        used.reserve(nnzTarget * 2);

        while (used.size() < nnzTarget) {
            int r = Random::get<int>(0, numRows - 1);
            int c = Random::get<int>(0, numCols - 1);
            auto key = pack(r, c);

            if (used.insert(key).second) {
                tmpRowIdx.push_back(r);
                tmpColIdx.push_back(c);
                tmpValues.push_back(Random::get<float>(minVal, maxVal));
            }
        }

        numNonzeros = tmpRowIdx.size();

        // Build rowPtrs
        rowPtrs.assign(numRows + 1, 0);
        for (std::size_t k = 0; k < numNonzeros; ++k) {
            ++rowPtrs[tmpRowIdx[k] + 1];
        }

        // Prefix sum
        for (int r = 0; r < numRows; ++r) {
            rowPtrs[r + 1] += rowPtrs[r];
        }

        colIdx.resize(numNonzeros);
        values.resize(numNonzeros);

        // Fill per-row positions
        std::vector<int> next(rowPtrs.begin(), rowPtrs.end());
        for (std::size_t k = 0; k < numNonzeros; ++k) {
            int r = tmpRowIdx[k];
            int pos = next[r]++;
            colIdx[pos] = tmpColIdx[k];
            values[pos] = tmpValues[k];
        }
    }
};

struct DeviceCRSMatrix {
    int numRows{};
    int numNonzeros{};
    const int* rowPtrs{};
    const int* colIdx{};
    const float* values{};
};

struct CRSMatrixDevice {
    DeviceCRSMatrix dev{};
    int* d_rowPtrs{nullptr};
    int* d_colIdx{nullptr};
    float* d_values{nullptr};

    CRSMatrixDevice() = default;
    explicit CRSMatrixDevice(const CRSMatrix& h);
    ~CRSMatrixDevice();

    CRSMatrixDevice(const CRSMatrixDevice&) = delete;
    CRSMatrixDevice& operator=(const CRSMatrixDevice&) = delete;
    CRSMatrixDevice(CRSMatrixDevice&& other) noexcept;
    CRSMatrixDevice& operator=(CRSMatrixDevice&& other) noexcept;
};
