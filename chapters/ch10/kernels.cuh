#pragma once

#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__global__ void simpleKernel(float* inputData, float* outputData);
__global__ void convergentKernel(float* inputData, float* outputData);
__global__ void convergentSharedMemKernel(float* inputData, float* outputData);
__global__ void segmentedKernel(const float* inputData, float* outputData, int length);
__global__ void coarsenedKernel(const float* inputData, float* outputData, int length);
__global__ void coarsenedKernelMax(const float* inputData, float* outputData, int length);
