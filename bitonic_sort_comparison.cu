#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define N 1024       // Number of elements in the array
#define BLOCK_SIZE 256 // CUDA block size

// CUDA kernel for Bitonic Sort
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

// CPU implementation of Bitonic Sort
void bitonicSortCPU(int *arr, int n) {
    int i, j, k;
    for (k = 2; k <= n; k <<= 1) {
        for (j = k >> 1; j > 0; j >>= 1) {
            for (i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0 && arr[i] > arr[ixj]) {
                        int temp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = temp;
                    }
                    if ((i & k) != 0 && arr[i] < arr[ixj]) {
                        int temp = arr[i];
                        arr[i] = arr[ixj];
                        arr[ixj] = temp;
                    }
                }
            }
        }
    }
}

// Helper function to fill the array with random values
void fillArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // Random values between 0 and 99
    }
}

// Helper function to check if the array is sorted
int isSorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    // Default values
    int n = N;
    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = 4;

    // Parse command-line arguments
    if (argc >= 2) n = atoi(argv[1]);               // First argument: number of elements
    if (argc >= 3) threadsPerBlock = atoi(argv[2]); // Second argument: threads per block
    if (argc >= 4) numBlocks = atoi(argv[3]);       // Third argument: number of blocks

    printf("Array Size: %d, Threads per Block: %d, Number of Blocks: %d\n", n, threadsPerBlock, numBlocks);

    int *arrCPU, *arrGPU, *d_arr;
    clock_t start, end;

    // Allocate memory on host
    arrCPU = (int *)malloc(n * sizeof(int));
    arrGPU = (int *)malloc(n * sizeof(int));

    // Fill array with random values
    fillArray(arrCPU, n);
    memcpy(arrGPU, arrCPU, n * sizeof(int)); // Copy values for GPU

    // CPU Bitonic Sort
    start = clock();
    bitonicSortCPU(arrCPU, n);
    end = clock();
    double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total time taken by the CPU part = %lf seconds\n", cpuTime);

    // Check if CPU sort is correct
    if (!isSorted(arrCPU, n)) {
        printf("CPU sorting failed!\n");
        return 1;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arrGPU, n * sizeof(int), cudaMemcpyHostToDevice);

    // GPU Bitonic Sort
    start = clock();
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<numBlocks, threadsPerBlock>>>(d_arr, n, j, k);
            cudaDeviceSynchronize();
        }
    }
    end = clock();
    double gpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Copy result back to host
    cudaMemcpy(arrGPU, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Check if GPU sort is correct
    if (!isSorted(arrGPU, n)) {
        printf("GPU sorting failed!\n");
        return 1;
    }

    // Calculate and print speedup
    double speedup = cpuTime / gpuTime;
    printf("GPU Time = %lf seconds\n", gpuTime);
    printf("Speedup (CPU/GPU) = %lfx\n", speedup);

    // Free memory
    cudaFree(d_arr);
    free(arrCPU);
    free(arrGPU);

    return 0;
}
