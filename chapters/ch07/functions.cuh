#pragma once

void conv2dCPU(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2d(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dConstMem(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dTiledIn(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dTiledOut(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
void conv2dTiledCached(const float *inArray, const float *filter, float *outArray, int radius, int height, int width);
