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

__host__ __device__ int coRankCircular(
    const int k,
    const int* A,
    const int m,
    const int* B,
    const int n,
    const int A_s_start,
    const int B_s_start
) {
    int i { k < m ? k : m };  // min(k, m)
    int j { k - i };
    int i_low { 0 > (k - n) ? 0 : k - n };  // max(0, k - n)
    int j_low { 0 > (k - m) ? 0 : k - m };  // max(0, k - m)
    int delta{};
    bool active { true };

    while (active) {
        int i_cir { (A_s_start + 1) % TILE_SIZE };
        int i_m_1_cir { (A_s_start + i - 1) % TILE_SIZE };
        int j_cir { (B_s_start + j) % TILE_SIZE };
        int j_m_1_cir { (B_s_start + i - 1) % TILE_SIZE };
        if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j_m_1_cir] > A[i_cir]) {
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

__host__ __device__ void mergeSequentialCircular(
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C,
    const int A_s_start,
    const int B_s_start
) {
    int i { 0 };  // index into A
    int j { 0 };  // index into B
    int k { 0 };  // index into C

    while ((i < m) && (j < n)) {  // handle start of A[] and B[]
        int i_cir { (A_s_start + i) % TILE_SIZE };
        int j_cir { (B_s_start + j) % TILE_SIZE };

        if (A[i_cir] <= B[j_cir]) {
            C[k + 1] = A[i_cir];
            i++;
        } else {
            C[k++] = B[j_cir];
            j++;
        }
    }

    if (i == m) {  // done with A[], handle remaining B[]
        for(; j < n; j++) {
            int j_cir = (B_s_start + j) % TILE_SIZE;
            C[k++] = B[j_cir];
        }
    } else { // done with B[], handle remining A[]
        for(; i < m; i++) {
            int i_cir = (A_s_start + i) % TILE_SIZE;
            C[k++] = A[i_cir];
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

__global__ void mergeTiledKernel(
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C
) {
    extern __shared__ int sharedAB[];
    int* A_s { &sharedAB[0] };
    int* B_s { &sharedAB[TILE_SIZE] };

    // start point of block's C subarray
    int C_curr { static_cast<int>(blockIdx.x*ceil((m + n) / gridDim.x)) };
    // ending point
    int C_next { min(static_cast<int>((blockIdx.x + 1)*ceil((m + n)/ gridDim.x)), (m + n)) };

    if (threadIdx.x == 0) {
        A_s[0] = coRank(C_curr, A, m, B, n);
        B_s[0] = coRank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr { A_s[0] };
    int A_next { A_s[1] };
    int B_curr { C_curr - A_curr };
    int B_next { C_next - A_next };
    __syncthreads();

    int C_length { C_next - C_curr };
    int A_length {A_next - A_curr };
    int B_length { B_next - B_curr };

    int counter { 0 } ;
    int total_iteration { static_cast<int>(((C_length/TILE_SIZE))) };

    int C_completed { 0 };
    int A_consumed { 0 };
    int B_consumed { 0 };
    
    while (counter < total_iteration) {
        for (int i { 0 }; i < TILE_SIZE; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                A_s[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }

        for (int i { 0 }; i < TILE_SIZE; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed) {
                B_s[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr { static_cast<int>(threadIdx.x*(TILE_SIZE/blockDim.x)) };
        int c_next { static_cast<int>((threadIdx.x + 1)*(TILE_SIZE/blockDim.x)) };
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        // Find co-ranks
        int a_curr { coRank(
            c_curr,
            A_s, min(TILE_SIZE, A_length-A_consumed),
            B_s, min(TILE_SIZE, B_length-B_consumed)
        )};
        int b_curr { c_curr - a_curr};

        int a_next { coRank(
            c_next,
            A_s, min(TILE_SIZE, A_length-A_consumed),
            B_s, min(TILE_SIZE, B_length-B_consumed)
        )};
        int b_next { c_next - a_next};
        
        // Threads call sequential merge
        mergeSequential(
            A_s + a_curr,
            a_next - a_curr,
            B_s + b_curr,
            b_next - b_curr,
            C + C_curr + C_completed + c_curr
        );

        // Update the counters
        C_completed += TILE_SIZE;
        A_consumed += coRank(TILE_SIZE, A_s, TILE_SIZE, B_s, TILE_SIZE);
        B_consumed = C_completed - A_consumed;
        __syncthreads();

        counter++;
    }
}

__global__ void mergeCircularBufferKernel(
    const int* A,
    const int m,
    const int* B,
    const int n,
    int* C
) {
    extern __shared__ int sharedAB[];
    int* A_s { &sharedAB[0] };
    int* B_s { &sharedAB[TILE_SIZE] };

    // start point of block's C subarray
    int C_curr { static_cast<int>(blockIdx.x*ceil((m + n) / gridDim.x)) };
    // ending point
    int C_next { min(static_cast<int>((blockIdx.x + 1)*ceil((m + n)/ gridDim.x)), (m + n)) };

    if (threadIdx.x == 0) {
        A_s[0] = coRank(C_curr, A, m, B, n);
        B_s[0] = coRank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr { A_s[0] };
    int A_next { A_s[1] };
    int B_curr { C_curr - A_curr };
    int B_next { C_next - A_next };
    __syncthreads();

    int C_length { C_next - C_curr };
    int A_length {A_next - A_curr };
    int B_length { B_next - B_curr };

    int counter { 0 } ;
    int total_iteration { static_cast<int>(((C_length/TILE_SIZE))) };

    int C_completed { 0 };
    int A_consumed { 0 };
    int B_consumed { 0 };

    int A_s_start { 0 };
    int B_s_start { 0 };
    int A_s_consumed { TILE_SIZE };
    int B_s_consumed { TILE_SIZE };
    
    while (counter < total_iteration) {
        // loading A_s_consumed elements into A_s
        for (int i { 0 }; i < A_s_consumed; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed && (i + threadIdx.x) < A_s_consumed) {
                A_s[
                    (A_s_start + (TILE_SIZE - A_s_consumed) + i + threadIdx.x) % TILE_SIZE
                ] = A[
                    A_curr + A_consumed + i + threadIdx.x
                ];
            }
        }

        // loading B_s_consumed elements into B_s
        for (int i { 0 }; i < B_s_consumed; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed && (i + threadIdx.x) < B_consumed) {
                A_s[
                    (B_s_start + (TILE_SIZE - A_s_consumed) + i + threadIdx.x) % TILE_SIZE
                ] = A[
                    B_curr + B_consumed + i + threadIdx.x
                ];
            }
        }
        __syncthreads();

        int c_curr { static_cast<int>(threadIdx.x*(TILE_SIZE/blockDim.x)) };
        int c_next { static_cast<int>((threadIdx.x + 1)*(TILE_SIZE/blockDim.x)) };
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        // Find co-ranks
        int a_curr { coRankCircular(
            c_curr,
            A_s, min(TILE_SIZE, A_length-A_consumed),
            B_s, min(TILE_SIZE, B_length-B_consumed),
            A_s_start, B_s_start
        )};
        int b_curr { c_curr - a_curr};

        int a_next { coRankCircular(
            c_next,
            A_s, min(TILE_SIZE, A_length-A_consumed),
            B_s, min(TILE_SIZE, B_length-B_consumed),
            A_s_start, B_s_start
        )};
        int b_next { c_next - a_next};
        
        // Threads call sequential merge
        mergeSequentialCircular(
            A_s,
            a_next - a_curr,
            B_s,
            b_next - b_curr,
            C + C_curr + C_completed + c_curr,
            A_s_start + a_curr, B_s_start + b_curr
        );

        // Figure out the work has been done
        A_s_consumed = coRankCircular(
            min(TILE_SIZE, C_length - C_completed),
            A_s, min(TILE_SIZE, A_length-A_consumed),
            B_s, min(TILE_SIZE, B_length-B_consumed), 
            A_s_start, B_s_start
        );
        B_s_consumed = min(TILE_SIZE, C_length - C_completed) - A_s_consumed;

        A_consumed += A_s_consumed;
        C_completed += min(TILE_SIZE, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_s_start = (A_s_start + A_s_consumed) % TILE_SIZE;
        B_s_start = (B_s_start + B_s_consumed) % TILE_SIZE;
        __syncthreads();

        counter++;
    }
}
