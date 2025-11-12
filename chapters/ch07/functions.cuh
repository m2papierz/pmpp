#pragma once

// 2D functions
void conv2dCPU(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2d(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dConstMem(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dTiledIn(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dTiledOut(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dTiledCached(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);

// 3D functions
void conv3dCPU(const float *inArray, const float *filter, float *outArray, int radius, int height, int width, int depth);
void conv3d(const float *inArray, const float *filter, float *outArray, int radius, int height, int width, int depth);
void conv3dConstMem(const float *inArray, const float *filter, float *outArray, int radius, int height, int width, int depth);
void conv3dTiled(const float *inArray, const float *filter, float *outArray, int radius, int height, int width, int depth);
