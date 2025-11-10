#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void conv2dKernel(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
