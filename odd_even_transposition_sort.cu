#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// CUDA kernel for parallel Bubble Sort (Odd-Even Transposition Sort)
__global__ void oddEvenTranspositionSort(int *arr, int n, int isOddPhase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int i = 2 * idx + (isOddPhase ? 1 : 0);

    if (i < n - 1 && arr[i] > arr[i + 1]) {
        int temp = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = temp;
    }
}

// CPU version of Bubble Sort
void bubbleSortCPU(int *arr, int n) {
    int phase, i;
    for (phase = 0; phase < n; phase++) {
        int isOddPhase = (phase % 2 == 1);
        for (i = (isOddPhase ? 1 : 0); i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
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
    int n = 1024;                // Number of elements
    int threadsPerBlock = 256;   // Threads per block
    int numBlocks = 4;           // Number of blocks

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

    // CPU Bubble Sort
    start = clock();
    bubbleSortCPU(arrCPU, n);
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

    // GPU Bubble Sort
    start = clock();
    for (int phase = 0; phase < n; phase++) {
        int isOddPhase = (phase % 2 == 1);
        oddEvenTranspositionSort<<<numBlocks, threadsPerBlock>>>(d_arr, n, isOddPhase);
        cudaDeviceSynchronize();
    }
    end = clock();
    double gpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Copy result back to host
    cudaMemcpy(arrGPU, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // // Print the sorted GPU array
    // printf("Sorted Array (GPU):\n");
    // for (int i = 0; i < n; i++) {
    //     printf("%d ", arrGPU[i]);
    //     if ((i + 1) % 10 == 0) { // Print 10 numbers per line for better readability
    //         printf("\n");
    //     }
    // }
    // printf("\n");

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
