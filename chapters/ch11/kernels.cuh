#pragma once

#define SECTION_SIZE 1024  // for Kogge-Stone & Brent-Kung kernels
#define COARSE_FACTOR 4    // for coarsened kernel

__global__ void koggeStoneKernel(const float* x, float* y, unsigned int n);
__global__ void koggeStoneKernelDoubleBuffer(const float* x, float* y, unsigned int n);
__global__ void brentKungKernel(const float* x, float* y, unsigned int n);
__global__ void coarsenedThreePhaseKernel(const float* x, float* y, unsigned int n);

// Hierarchical scan kernels
__global__ void hierarchicalKernelPhase1(float* x, float* y, float* s, unsigned int n);
__global__ void hierarchicalKernelPhase2(float* s, unsigned int n);
__global__ void hierarchicalKernelPhase3(float* y, float* s, unsigned int n);
__global__ void hierarchicalDominoKernel(float* x, float* y, float* scanValue, int* flags, int* blockCounter, unsigned int n);
