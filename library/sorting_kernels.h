#ifndef SORTING_KERNELS_H
#define SORTING_KERNELS_H

#include <cuda.h>

// Kernel declarations
__global__ void sample_sort(int *A, int N);
__global__ void merge_chunks(int *A, int N, int chunk_size);
__global__ void bitonicSortKernel(int *arr, int n, int j, int k);

// Combined function for Sample Sort and Global Merge
void combined_sample_sort_and_merge(int *A, int N);

#endif // SORTING_KERNELS_H
