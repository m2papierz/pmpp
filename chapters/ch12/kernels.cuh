#pragma once

#define BLOCK_SIZE 512
#define TILE_SIZE 256

__host__ __device__ void mergeSequential(const int* A, const int m, const int* B, const int n, int* C);
__global__ void mergeBasicKernel(const int* A, const int m, const int* B, const int n, int* C);
__global__ void mergeTiledKernel(const int* A, const int m, const int* B, const int n, int* C);
__global__ void mergeCircularBufferKernel(const int* A, const int m, const int* B, const int n, int* C);
