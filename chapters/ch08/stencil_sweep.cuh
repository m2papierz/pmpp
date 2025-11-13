#pragma once

#define OUT_TILE_DIM 8                         // for navie, and shared memory versions
#define IN_TILE_DIM (OUT_TILE_DIM + 2)         // for naive, and shared memory versions
#define OUT_TILE_DIM_TC 30                     // for thread coarsening version     
#define IN_TILE_DIM_TC (OUT_TILE_DIM_TC + 2)   // for thread coarsening version
#define OUT_TILE_DIM_RT 30                     // for register tiling version     
#define IN_TILE_DIM_RT (OUT_TILE_DIM_RT + 2)   // for register tiling version

void stencilCPU(float* in, float* out, unsigned int n, int c0, int c1, int c2, int c3, int c4, int c5, int c6);
void stencilNaive(float* in, float* out, unsigned int n, int c0, int c1, int c2, int c3, int c4, int c5, int c6);
void stencilSharedMem(float* in, float* out, unsigned int n, int c0, int c1, int c2, int c3, int c4, int c5, int c6);
void stencilThreadCoarsening(float* in, float* out, unsigned int n, int c0, int c1, int c2, int c3, int c4, int c5, int c6);
void stencilRegisterTiling(float* in, float* out, unsigned int n, int c0, int c1, int c2, int c3, int c4, int c5, int c6);
