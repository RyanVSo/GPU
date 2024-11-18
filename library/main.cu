#include <stdio.h>
#include <stdlib.h>
#include "sorting_kernels.h"

#define ARRAY_SIZE 1024  // Size of the array for sorting

void fillArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;  // Random values between 0 and 99
    }
}

void printArray(const char *label, int *arr, int n) {
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void callBitonicSort(int *A, int N) {
    int *d_A;
    cudaMalloc((void **)&d_A, N * sizeof(int));
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<numBlocks, threadsPerBlock>>>(d_A, N, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
}

int main() {
    int N = ARRAY_SIZE;
    int *A = (int *)malloc(N * sizeof(int));
    int *B = (int *)malloc(N * sizeof(int));

    fillArray(A, N);
    fillArray(B, N);

    printf("Calling Combined Sample Sort and Global Merge:\n");
    printf("Before Sorting:\n");
    printArray("Array A", A, N);

    combined_sample_sort_and_merge(A, N);

    printf("After Sorting:\n");
    printArray("Array A", A, N);

    printf("Calling Bitonic Sort:\n");
    printf("Before Sorting:\n");
    printArray("Array B", B, N);

    callBitonicSort(B, N);

    printf("After Sorting:\n");
    printArray("Array B", B, N);

    free(A);
    free(B);

    return 0;
}
