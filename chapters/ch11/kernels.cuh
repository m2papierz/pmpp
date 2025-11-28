#pragma once

#define SECTION_SIZE 1024

__global__ void koggeStoneKernel(const float* x, float* y, unsigned int n);
__global__ void koggeStoneKernelDoubleBuffer(const float* x, float* y, unsigned int n);
__global__ void brentKungKernel(const float* x, float* y, unsigned int n);
