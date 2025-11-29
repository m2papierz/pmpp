#pragma once

void scanSequential(const float* x, float* y, unsigned int n);
void koggeStone(const float* x, float* y, unsigned int n);
void koggeStoneDoubleBuffer(const float* x, float* y, unsigned int n);
void brentKung(const float* x, float* y, unsigned int n);
void coarsenedThreePhase(const float* x, float* y, unsigned int n);
