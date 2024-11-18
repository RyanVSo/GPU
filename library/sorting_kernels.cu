#include "sorting_kernels.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PER_BLOCK 128  // Fixed number of threads per block

// Sample Sort kernel
__global__ void sample_sort(int *A, int N) {
    __shared__ int loc[PER_BLOCK];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;  // Avoid out-of-bounds access
    int k = threadIdx.x;
    loc[k] = A[i];
    __syncthreads();

    for (int j = 0; j < PER_BLOCK / 2; j++) {
        if (k % 2 == 0 && k < PER_BLOCK - 1) {
            if (loc[k] > loc[k + 1]) {
                int temp = loc[k];
                loc[k] = loc[k + 1];
                loc[k + 1] = temp;
            }
        }
        __syncthreads();

        if (k % 2 == 1 && k < PER_BLOCK - 1) {
            if (loc[k] > loc[k + 1]) {
                int temp = loc[k];
                loc[k] = loc[k + 1];
                loc[k + 1] = temp;
            }
        }
        __syncthreads();
    }

    A[i] = loc[k];
}

// Merge Chunks kernel
__global__ void merge_chunks(int *A, int N, int chunk_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int start1 = 2 * idx * chunk_size;
    int start2 = start1 + chunk_size;

    if (start1 >= N || start2 >= N) return;

    int end1 = min(start2, N);
    int end2 = min(start2 + chunk_size, N);

    int *temp = new int[end2 - start1];
    int i = start1, j = start2, k = 0;

    while (i < end1 && j < end2) {
        if (A[i] <= A[j]) {
            temp[k++] = A[i++];
        } else {
            temp[k++] = A[j++];
        }
    }
    while (i < end1) temp[k++] = A[i++];
    while (j < end2) temp[k++] = A[j++];

    for (i = 0; i < k; i++) {
        A[start1 + i] = temp[i];
    }

    delete[] temp;
}

// Combined Sample Sort and Merge Chunks
void combined_sample_sort_and_merge(int *A, int N) {
    int num_blocks = (int)ceil((float)N / PER_BLOCK);

    int *d_A;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads_per_block(PER_BLOCK);
    dim3 num_of_blocks(num_blocks);
    sample_sort<<<num_of_blocks, threads_per_block>>>(d_A, N);
    cudaDeviceSynchronize();

    int chunk_size = PER_BLOCK;

    while (chunk_size < N) {
        int num_chunks = (N + 2 * chunk_size - 1) / (2 * chunk_size);
        merge_chunks<<<num_chunks, threads_per_block>>>(d_A, N, chunk_size);
        cudaDeviceSynchronize();
        chunk_size *= 2;
    }

    cudaMemcpy(A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}

// Bitonic Sort kernel
__global__ void bitonicSortKernel(int *arr, int n, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx;

    if (i < n) {
        int ixj = i ^ j;

        if (ixj > i) {
            if ((i & k) == 0) {
                if (arr[i] > arr[ixj]) {
                    int temp = arr[i];
                    arr[i] = arr[ixj];
                    arr[ixj] = temp;
                }
            }
            if ((i & k) != 0) {
                if (arr[i] < arr[ixj]) {
                    int temp = arr[i];
                    arr[i] = arr[ixj];
                    arr[ixj] = temp;
                }
            }
        }
    }
}
