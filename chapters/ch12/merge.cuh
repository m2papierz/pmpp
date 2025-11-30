#pragma once

#include "kernels.cuh"  // for mergeSequential

void mergeBasic(const int* A, const int m, const int* B, const int n, int* C);
