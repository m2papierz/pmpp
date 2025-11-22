#pragma once

// CPU serial implementations
void reductionSumSerial(const float* inputData, float* outputData, int n);
void reductionMaxSerial(const float* inputData, float* outputData, int n);

// GPU implementations
void reductionSimple(float* inputData, float* outputData, int n);
void reductionConvergent(float* inputData, float* outputData, int n);
void reductionConvergentSharedMem(float* inputData, float* outputData, int n);
void reductionSegmented(const float* inputData, float* outputData, int n);
void reductionCoarsed(const float* inputData, float* outputData, int n);
void reductionMaxCoarsed(const float* inputData, float* outputData, int n);
