#pragma once

void histogramSequential(const char* data, unsigned int length, unsigned int* histo);
void histogramNaive(const char* data, unsigned int length, unsigned int* histo);
void histogramPrivate(const char* data, unsigned int length, unsigned int* histo);
void histogramSharedMem(const char* data, unsigned int length, unsigned int* histo);
void histogramContiguousPart(const char* data, unsigned int length, unsigned int* histo);
void histogramContiguousInter(const char* data, unsigned int length, unsigned int* histo);
void histogramAggregation(const char* data, unsigned int length, unsigned int* histo);
