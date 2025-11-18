#pragma once

void histogramSequential(const char* data, unsigned int length, unsigned int* histo);
void histogramNaive(const char* data, unsigned int length, unsigned int* histo);
void histogramPrivate(const char* data, unsigned int length, unsigned int* histo);
void histogramSharedMem(const char* data, unsigned int length, unsigned int* histo);
