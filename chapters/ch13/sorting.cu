#include "sorting.cuh"
#include "kernels/radix.cuh"
#include "kernels/merge.cuh"
#include "utils.hpp"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
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

void radixSortCoalescedMultibit(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int n
) {
    unsigned int radix { 1u << RADIX_BITS };

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(n, BLOCK_SIZE));
    unsigned int numBlocks { gridSize.x };

    std::size_t sizeArr { static_cast<std::size_t>(n)*sizeof(unsigned int) };
    std::size_t sizePerBucket { static_cast<std::size_t>(numBlocks)*sizeof(unsigned int) };
    std::size_t sizeBlockBuckets{ static_cast<std::size_t>(radix)*sizePerBucket };

    // Device buffers
    unsigned int* inputArr_d { nullptr };
    unsigned int* outputArr_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputArr_d,  sizeArr));
    CUDA_CHECK(cudaMalloc(&outputArr_d, sizeArr));

    unsigned int* localScan_d { nullptr };
    CUDA_CHECK(cudaMalloc(&localScan_d, sizeArr));

    unsigned int* blockBucketCounts_d { nullptr };  // [radix][numBlocks]
    unsigned int* blockBucketOffsets_d { nullptr }; // [radix][numBlocks]
    CUDA_CHECK(cudaMalloc(&blockBucketCounts_d, sizeBlockBuckets));
    CUDA_CHECK(cudaMalloc(&blockBucketOffsets_d, sizeBlockBuckets));

    unsigned int* bucketTotals_d { nullptr }; // [radix]
    unsigned int* bucketStarts_d { nullptr }; // [radix]
    CUDA_CHECK(cudaMalloc(&bucketTotals_d, radix*sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&bucketStarts_d, radix*sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(inputArr_d, inputArr, sizeArr, cudaMemcpyHostToDevice));

    // Process RADIX_BITS bits per pass
    for (unsigned int shift { 0 }; shift < NUM_BITS; shift += RADIX_BITS) {
        // Reset bucket totals
        CUDA_CHECK(cudaMemset(bucketTotals_d, 0, radix*sizeof(unsigned int)));

        // 1) Local bucket counts + local indices in each block
        localSortMultibitKernel<<<gridSize, blockSize>>>(
            inputArr_d,
            localScan_d,
            blockBucketCounts_d,
            bucketTotals_d,
            n,
            shift,
            numBlocks,
            radix
        );
        CUDA_CHECK(cudaGetLastError());

        // 2) Exclusive scan over blocks for each bucket -> per-block bucket offsets
        for (unsigned int b { 0 }; b < radix; ++b) {
            unsigned int* countsBegin  { blockBucketCounts_d   + b*numBlocks };
            unsigned int* offsetsBegin { blockBucketOffsets_d  + b*numBlocks };

            thrust::exclusive_scan(
                thrust::device,
                countsBegin,
                countsBegin + numBlocks,
                offsetsBegin
            );
        }

        // 3) Exclusive scan over bucketTotals -> global bucket starts
        thrust::exclusive_scan(
            thrust::device,
            bucketTotals_d,
            bucketTotals_d + radix,
            bucketStarts_d
        );

        // 4) Scatter using stable indices
        scatterCoalescedMultibitKernel<<<gridSize, blockSize>>>(
            inputArr_d,
            outputArr_d,
            localScan_d,
            blockBucketOffsets_d,
            bucketStarts_d,
            n,
            shift,
            numBlocks,
            radix
        );
        CUDA_CHECK(cudaGetLastError());

        // 5) Swap buffers for next pass
        std::swap(inputArr_d, outputArr_d);
    }

    // Copy back (inputArr_d holds the final result after last swap)
    CUDA_CHECK(cudaMemcpy(outputArr, inputArr_d, sizeArr, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(inputArr_d));
    CUDA_CHECK(cudaFree(outputArr_d));
    CUDA_CHECK(cudaFree(localScan_d));
    CUDA_CHECK(cudaFree(blockBucketCounts_d));
    CUDA_CHECK(cudaFree(blockBucketOffsets_d));
    CUDA_CHECK(cudaFree(bucketTotals_d));
    CUDA_CHECK(cudaFree(bucketStarts_d));
}

void radixSortCoalescedCoarse(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int n
) {
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(utils::cdiv(n, BLOCK_SIZE * COARSE_FACTOR));
    unsigned int numBlocks { gridSize.x };

    std::size_t sizeArr { static_cast<std::size_t>(n) * sizeof(unsigned int) };
    std::size_t sizeBlockCounts { static_cast<std::size_t>(numBlocks) * sizeof(unsigned int) };

    unsigned int* inputArr_d { nullptr };
    unsigned int* outputArr_d { nullptr };
    CUDA_CHECK(cudaMalloc(&inputArr_d,  sizeArr));
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

    for (unsigned int iteration = 0; iteration < NUM_BITS; ++iteration) {
        // 1) Local scan (coarsened)
        localSortCoarseKernel<<<gridSize, blockSize>>>(
            inputArr_d,
            localScan_d,
            blockZeros_d,
            blockOnes_d,
            n,
            iteration
        );
        CUDA_CHECK(cudaGetLastError());

        // 2) Total zeros across blocks
        unsigned int totalZeros =
            thrust::reduce(
                thrust::device,
                blockZeros_d,
                blockZeros_d + numBlocks,
                0u,
                thrust::plus<unsigned int>()
            );

        // 3) Prefix over block zero counts
        thrust::exclusive_scan(
            thrust::device,
            blockZeros_d,
            blockZeros_d + numBlocks,
            blockZerosOffset_d
        );

        // 4) Prefix over block one counts
        thrust::exclusive_scan(
            thrust::device,
            blockOnes_d,
            blockOnes_d + numBlocks,
            blockOnesOffset_d
        );

        // 5) Scatter (coarsened)
        scatterCoalescedCoarseKernel<<<gridSize, blockSize>>>(
            inputArr_d,
            outputArr_d,
            localScan_d,
            blockZerosOffset_d,
            blockOnesOffset_d,
            totalZeros,
            n,
            iteration
        );
        CUDA_CHECK(cudaGetLastError());

        // 6) Swap for next pass
        std::swap(inputArr_d, outputArr_d);
    }

    CUDA_CHECK(cudaMemcpy(outputArr, inputArr_d, sizeArr, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(inputArr_d));
    CUDA_CHECK(cudaFree(outputArr_d));
    CUDA_CHECK(cudaFree(localScan_d));
    CUDA_CHECK(cudaFree(blockZeros_d));
    CUDA_CHECK(cudaFree(blockOnes_d));
    CUDA_CHECK(cudaFree(blockZerosOffset_d));
    CUDA_CHECK(cudaFree(blockOnesOffset_d));
}

void mergeSort(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int n
) {
    std::size_t sizeArr { static_cast<std::size_t>(n)*sizeof(unsigned int) };

    unsigned int* d_src { nullptr };
    unsigned int* d_dst { nullptr };
    CUDA_CHECK(cudaMalloc(&d_src, sizeArr));
    CUDA_CHECK(cudaMalloc(&d_dst, sizeArr));

    CUDA_CHECK(cudaMemcpy(d_src, inputArr, sizeArr, cudaMemcpyHostToDevice));

    // Bottom-up iterative merge sort on device
    for (unsigned int width = 1; width < n; width *= 2) {
        unsigned int numMerges = (n + 2 * width - 1) / (2 * width);  // ceil(n / (2*width))

        dim3 blockSize(BLOCK_SIZE);
        dim3 gridSize(numMerges);

        mergeRangesKernel<<<gridSize, blockSize>>>(
            d_src,
            d_dst,
            n,
            width
        );
        CUDA_CHECK(cudaGetLastError());

        // For next pass, swap src and dst
        std::swap(d_src, d_dst);
    }

    // After the last swap, d_src holds the sorted data
    CUDA_CHECK(cudaMemcpy(outputArr, d_src, sizeArr, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}


void thrustSortGPU(
    const unsigned int* inputArr,
    unsigned int* outputArr,
    const unsigned int n
) {
    std::size_t sizeArr { static_cast<std::size_t>(n) * sizeof(unsigned int) };

    unsigned int* d_arr { nullptr };
    CUDA_CHECK(cudaMalloc(&d_arr, sizeArr));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_arr, inputArr, sizeArr, cudaMemcpyHostToDevice));

    // Wrap raw pointer with thrust device_ptr
    thrust::device_ptr<unsigned int> d_begin(d_arr);
    thrust::device_ptr<unsigned int> d_end = d_begin + n;

    // Best Thrust device sort (radix-based for unsigned ints)
    thrust::sort(thrust::device, d_begin, d_end);

    // Copy sorted result back
    CUDA_CHECK(cudaMemcpy(outputArr, d_arr, sizeArr, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_arr));
}
