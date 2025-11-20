#pragma once

#define BIN_SIZE 4
#define NUM_BINS ((26 + BIN_SIZE - 1) / BIN_SIZE)
#define BLOCK_SIZE 512
#define CFACTOR 32

__global__ void histogramKernel(const char* data, unsigned int length, unsigned int* histo);
__global__ void histogramKernelPrivate(const char* data, unsigned int length, unsigned int* histo);
__global__ void histogramKernelSharedMem(const char* data, unsigned int length, unsigned int* histo);
__global__ void histogramKernelContiguousPart(const char* data, unsigned int length, unsigned int* histo);
__global__ void histogramKernelContiguousInter(const char* data, unsigned int length, unsigned int* histo);
__global__ void histogramKernelAggregation(const char* data, unsigned int length, unsigned int* histo);
