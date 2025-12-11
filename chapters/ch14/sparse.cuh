#pragma once

#include "kernels.cuh"

void spmv_coo_cpu(const COOMatrix& cooMatrix, const  float* x, float* y);
void spmv_coo(const COOMatrix& cooMatrix, const  float* x, float* y);
void spmv_crs_cpu(const CRSMatrix& crsMatrix, const float* x, float* y);
void spmv_crs(const CRSMatrix& crsMatrix, const float* x, float* y);
