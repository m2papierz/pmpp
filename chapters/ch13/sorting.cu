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

    std::size_t sizeInput { static_cast<std::size_t>(n*sizeof(unsigned int)) };
    std::size_t sizeOutput { static_cast<std::size_t>(n*sizeof(unsigned int)) };
    std::size_t sizeBits { static_cast<std::size_t>((n + 1)*sizeof(unsigned int)) };

    // Sort part arrays
    unsigned int* inputArr_d { nullptr };
    unsigned int* outputArr_d { nullptr };
    unsigned int* bits_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputArr_d, sizeInput));
    CUDA_CHECK(cudaMalloc(&outputArr_d, sizeOutput));
    CUDA_CHECK(cudaMalloc(&bits_d, sizeBits));

    CUDA_CHECK(cudaMemcpy(inputArr_d, inputArr, sizeInput, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(bits_d, 0, sizeBits));

    for (int iteration { 0 }; iteration < NUM_BITS; ++iteration) {
        // 1) compute bits[]
        computeBitsKernel<<<gridSize, blockSize>>>(inputArr_d, bits_d, n, iteration);
        CUDA_CHECK(cudaGetLastError());

        // 2) exclusive scan on bits with Thrust
        CUDA_CHECK(cudaMemset(bits_d + n, 0, sizeof(unsigned int)));
        thrust::exclusive_scan(thrust::device, bits_d, bits_d + (n + 1), bits_d);

        // 3) scatter using scanned bits
        scatterKernel<<<gridSize, blockSize>>>(inputArr_d, outputArr_d, bits_d, n, iteration);
        CUDA_CHECK(cudaGetLastError());

        // 4) Swap output with input 
        std::swap(inputArr_d, outputArr_d);
    }

    CUDA_CHECK(cudaMemcpy(outputArr, inputArr_d, sizeOutput, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputArr_d));
    CUDA_CHECK(cudaFree(outputArr_d));
    CUDA_CHECK(cudaFree(bits_d));
}
