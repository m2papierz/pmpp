#include "scan.cuh"
#include "utils.hpp"

void scanSequential(const float* x, float* y, unsigned int n) {
    y[0] = x[0];
    for(int i { 1 }; i < n; ++i) {
        y[i] = y[i - 1] + x[i];
    }
}
