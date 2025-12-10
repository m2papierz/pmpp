#include "merge.cuh"

__host__ __device__ unsigned int coRank(
    const unsigned int k,
    const unsigned int* A,
    const unsigned int m,
    const unsigned int* B,
    const unsigned int n
) {
    unsigned int i = (k < m) ? k : m;  // min(k, m)
    unsigned int j = k - i;

    // max(0, k - n) and max(0, k - m) without underflow
    unsigned int i_low = (k > n) ? (k - n) : 0;
    unsigned int j_low = (k > m) ? (k - m) : 0;

    unsigned int delta{};
    bool active = true;

    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

__host__ __device__ void mergeSequential(
    const unsigned int* A,
    const unsigned int m,
    const unsigned int* B,
    const unsigned int n,
    unsigned int* C
) {
    unsigned int i { 0 };  // index into A
    unsigned int j { 0 };  // index into B
    unsigned int k { 0 };  // index into C

    // merge main part
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }

    // remaining A
    while (i < m) {
        C[k++] = A[i++];
    }

    // remaining B
    while (j < n) {
        C[k++] = B[j++];
    }
}

__global__ void mergeRangesKernel(
    const unsigned int* src,
    unsigned int* dst,
    unsigned int n,
    unsigned int width
) {
    unsigned int mergeId { blockIdx.x };
    unsigned int left { mergeId * 2 * width };
    if (left >= n) return;

    unsigned int mid { min(left + width, n) };
    unsigned int right { min(left + 2 * width, n) };

    unsigned int m { mid - left };   // length of A
    unsigned int nn { right - mid };   // length of B

    const unsigned int* A { src + left };
    const unsigned int* B { src + mid };
    unsigned int* C { dst + left };

    unsigned int total { m + nn };
    if (total == 0) return;

    unsigned int tid { threadIdx.x };
    unsigned int nThreads { blockDim.x };

    unsigned int itemsPerThread { (total + nThreads - 1) / nThreads }; // ceil
    unsigned int k_start { min(tid * itemsPerThread, total) };
    unsigned int k_end { min(k_start + itemsPerThread, total) };

    if (k_start >= k_end) return;

    unsigned int a_start { coRank(k_start, A, m, B, nn) };
    unsigned int a_end { coRank(k_end,   A, m, B, nn) };
    unsigned int b_start { k_start - a_start };
    unsigned int b_end { k_end - a_end };

    mergeSequential(
        A + a_start, a_end - a_start,
        B + b_start, b_end - b_start,
        C + k_start
    );
}
