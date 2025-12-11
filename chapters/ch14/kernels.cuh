#pragma once

#include "formats.cuh"

#define BLOCK_SIZE 1024

__global__ void spmv_coo_kernel(
    DeviceCOOMatrix cooMatrix,
    const float* __restrict__ x,
    float* __restrict__ y
);
