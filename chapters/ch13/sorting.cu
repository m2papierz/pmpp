#include "sorting.cuh"
#include "kernels/radix.cuh"
#include "utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

void radixSortCPU(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int n
) {
    // Ping-pong buffers to mirror device in/out arrays
    std::vector<unsigned int> in(inputArr, inputArr + n);
    std::vector<unsigned int> out(n);

    // bits has size n + 1 so bits[n] can hold the total number of ones
    std::vector<unsigned int> bits(n + 1);

    for (unsigned int iteration { 0 }; iteration < NUM_BITS; ++iteration) {
        // 1) Extract current bit for each element
        for (unsigned int i { 0 }; i < n; ++i) {
            unsigned int key { in[i] };
            bits[i] = (key >> iteration) & 1u;
        }

        // dummy element so exclusive_scan puts total ones into bits[n]
        bits[n] = 0;

        // 2) Exclusive scan on bits[0..n] in-place:
        thrust::exclusive_scan(thrust::host, bits.begin(), bits.end(), bits.begin());

        // 3) scatter
        unsigned int numOnesTotal { bits[n] };
        for (unsigned int i { 0 }; i < n; ++i) {
            unsigned int key = in[i];
            unsigned int bit = (key >> iteration) & 1u;
            unsigned int numOnesBefore = bits[i];

            unsigned int destination {
                (bit == 0)
                    ? (i - numOnesBefore)
                    : (n - numOnesTotal + numOnesBefore)
            };

            out[destination] = key;
        }

        // 4) Next iteration uses 'out' as input
        in.swap(out);
    }

    // After all iterations, the sorted data is in 'in'
    std::copy(in.begin(), in.end(), outputArr);
}

void radixSort(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int n
) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(n, BLOCK_SIZE));

    std::size_t sizeArr { static_cast<std::size_t>(n*sizeof(unsigned int)) };
    std::size_t sizeBits { static_cast<std::size_t>((n + 1)*sizeof(unsigned int)) };

    // Sort part arrays
    unsigned int* inputArr_d { nullptr };
    unsigned int* outputArr_d { nullptr };
    unsigned int* bits_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputArr_d, sizeArr));
    CUDA_CHECK(cudaMalloc(&outputArr_d, sizeArr));
    CUDA_CHECK(cudaMalloc(&bits_d, sizeBits));

    CUDA_CHECK(cudaMemcpy(inputArr_d, inputArr, sizeArr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(bits_d, 0, sizeBits));

    for (int iteration { 0 }; iteration < NUM_BITS; ++iteration) {
        // Compute bits[]
        computeBitsKernel<<<gridSize, blockSize>>>(inputArr_d, bits_d, n, iteration);
        CUDA_CHECK(cudaGetLastError());

        // Exclusive scan on bits with Thrust
        CUDA_CHECK(cudaMemset(bits_d + n, 0, sizeof(unsigned int)));
        thrust::exclusive_scan(thrust::device, bits_d, bits_d + (n + 1), bits_d);

        // Scatter using scanned bits
        scatterKernel<<<gridSize, blockSize>>>(inputArr_d, outputArr_d, bits_d, n, iteration);
        CUDA_CHECK(cudaGetLastError());

        // Swap output with input 
        std::swap(inputArr_d, outputArr_d);
    }

    CUDA_CHECK(cudaMemcpy(outputArr, inputArr_d, sizeArr, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputArr_d));
    CUDA_CHECK(cudaFree(outputArr_d));
    CUDA_CHECK(cudaFree(bits_d));
}

void radixSortCoalesced(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int n
) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(n, BLOCK_SIZE));
    unsigned int numBlocks { gridSize.x };

    std::size_t sizeArr { static_cast<std::size_t>(n*sizeof(unsigned int)) };
    std::size_t sizeBlockCounts { static_cast<std::size_t>(numBlocks*sizeof(unsigned int)) };

    unsigned int* inputArr_d { nullptr };
    unsigned int* outputArr_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputArr_d, sizeArr));
    CUDA_CHECK(cudaMalloc(&outputArr_d, sizeArr));

    unsigned int* localScan_d { nullptr };
    CUDA_CHECK(cudaMalloc(&localScan_d, sizeArr));

    unsigned int* blockZeros_d { nullptr };
    unsigned int* blockOnes_d { nullptr };
    CUDA_CHECK(cudaMalloc(&blockZeros_d, sizeBlockCounts));
    CUDA_CHECK(cudaMalloc(&blockOnes_d, sizeBlockCounts));

    unsigned int* blockZerosOffset_d { nullptr };
    unsigned int* blockOnesOffset_d { nullptr };
    CUDA_CHECK(cudaMalloc(&blockZerosOffset_d, sizeBlockCounts));
    CUDA_CHECK(cudaMalloc(&blockOnesOffset_d, sizeBlockCounts));

    CUDA_CHECK(cudaMemcpy(inputArr_d, inputArr, sizeArr, cudaMemcpyHostToDevice));

    for (int iteration { 0 }; iteration < NUM_BITS; ++iteration) {
        // Local sort
        localSortKernel<<<gridSize, blockSize>>>(
            inputArr_d,
            localScan_d,
            blockZeros_d,
            blockOnes_d,
            n, iteration
        );
        CUDA_CHECK(cudaGetLastError());

        unsigned int totalZeros {
            thrust::reduce(
                thrust::device,
                blockZeros_d,
                blockZeros_d + numBlocks,
                0u, thrust::plus<unsigned int>()
            )
        };

        // Exclusive scan over zeros
        thrust::exclusive_scan(
            thrust::device,
            blockZeros_d,
            blockZeros_d  + numBlocks,
            blockZerosOffset_d
        );

        // Exclusive scan over zeros
        thrust::exclusive_scan(
            thrust::device,
            blockOnes_d,
            blockOnes_d + numBlocks,
            blockOnesOffset_d
        );

        // 3) Scatter globally
        scatterCoalescedKernel<<<gridSize, blockSize>>>(
            inputArr_d,
            outputArr_d,
            localScan_d,
            blockZerosOffset_d,
            blockOnesOffset_d,
            totalZeros,
            n, iteration
        );
        CUDA_CHECK(cudaGetLastError());

        // Swap output with input 
        std::swap(inputArr_d, outputArr_d);
    }

    CUDA_CHECK(cudaMemcpy(outputArr, inputArr_d, sizeArr, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputArr_d));
    CUDA_CHECK(cudaFree(localScan_d));
    CUDA_CHECK(cudaFree(blockZeros_d));
    CUDA_CHECK(cudaFree(blockOnes_d));
    CUDA_CHECK(cudaFree(blockZerosOffset_d));
    CUDA_CHECK(cudaFree(blockOnesOffset_d));
    CUDA_CHECK(cudaFree(outputArr_d));
}
