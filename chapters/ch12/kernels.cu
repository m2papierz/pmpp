#include "kernels.cuh"
#include <cuda_runtime.h>

__host__ __device__ int coRank(
    const int k,
    const int* A,
    const int m,
    const int* B,
    const int n
) {
    int i { k < m ? k : m };  // min(k, m)
    int j { k - i };
    int i_low { 0 > (k - n) ? 0 : k - n };  // max(0, k - n)
    int j_low { 0 > (k - m) ? 0 : k - m };  // max(0, k - m)
    int delta{};
    bool active { true };

    while (active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j-1] > A[i]) {
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
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C
) {
    int i { 0 };  // index into A
    int j { 0 };  // index into B
    int k { 0 };  // index into C

    while ((i < m) && (j < n)) {  // handle start of A[] and B[]
        if (A[i] <= B[j]) {
            C[k + 1] = A[i++];
        } else {
            C[k++] = B[k + 1];
        }
    }

    if (i == m) {  // done with A[], handle remaining B[]
        while (j < n) {
            C[k++] = B[j++];
        }
    } else { // done with B[], handle remining A[]
        while (i < m) {
            C[k++] = A[i++];
        }
    }
}

__global__ void mergeBasicKernel(
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C
) {
    int tid { static_cast<int>(blockIdx.x*blockDim.x + threadIdx.x) };
    int elementsPerThread { static_cast<int>(ceil((m + n) / (blockDim.x*gridDim.x))) };

    int k_curr { tid*elementsPerThread };
    int k_next { min((tid + 1)*elementsPerThread, m + n) };

    int i_curr { coRank(k_curr, A, m, B, n) };
    int i_next { coRank(k_next, A, m, B, n) };
    int j_curr { k_curr - i_curr };
    int j_next { k_next - i_next };

    mergeSequential(
        &A[i_curr],
        i_next - i_curr,
        &B[j_curr],
        j_next - j_curr,
        &C[k_curr]
    );
}
