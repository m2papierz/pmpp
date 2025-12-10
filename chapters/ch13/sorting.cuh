#pragma once

void radixSortCPU(const unsigned int* inputArr, unsigned int* outputArr, const unsigned int n);
void radixSort(const unsigned int* inputArr, unsigned int* outputArr, const unsigned int n);
void radixSortCoalesced(const unsigned int* inputArr, unsigned int* outputArr, const unsigned int n);
void radixSortCoalescedMultibit(const unsigned int* inputArr, unsigned int* outputArr, const unsigned int n);
void radixSortCoalescedCoarse(const unsigned int* inputArr, unsigned int* outputArr, const unsigned int n);
void mergeSort(const unsigned int* inputArr, unsigned int* outputArr, const unsigned int n);
