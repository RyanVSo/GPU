#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>

#define PER_BLOCK 64 // Fixed number of threads per block

__global__ void sample_sort(int *A, int N) {
    __shared__ int loc[PER_BLOCK];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = threadIdx.x;

    // Initialize shared memory to avoid undefined values
    loc[k] = (i < N) ? A[i] : INT_MAX; // Use a sentinel value for out-of-bounds indices
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

    // Write back to global memory only if within bounds
    if (i < N) {
        A[i] = loc[k];
    }
}

__global__ void merge_chunks(int *A, int N, int chunk_size, int *temp_global) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the start of the two chunks to merge
    int start1 = 2 * idx * chunk_size;
    int start2 = start1 + chunk_size;

    if (start1 >= N || start2 >= N) return;

    int end1 = min(start2, N);
    int end2 = min(start2 + chunk_size, N);

    // Use a portion of the pre-allocated temporary array for this block
    int *temp = &temp_global[start1];

    int i = start1, j = start2, k = 0;

    // Merge the two chunks
    while (i < end1 && j < end2) {
        if (A[i] <= A[j]) {
            temp[k++] = A[i++];
        } else {
            temp[k++] = A[j++];
        }
    }
    while (i < end1) temp[k++] = A[i++];
    while (j < end2) temp[k++] = A[j++];

    // Copy merged result back to the array
    for (i = 0; i < k; i++) {
        A[start1 + i] = temp[i];
    }
}



void merge(int *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void mergeSort(int *arr, int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;
        mergeSort(arr, left, middle);
        mergeSort(arr, middle + 1, right);
        merge(arr, left, middle, right);
    }
}

void printArray(const char *label, int *arr, int size) {
    printf("%s: ", label);
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void printFullArray(const char *label, int *arr, int size) {
    printf("%s:\n", label);
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
        if ((i + 1) % 10 == 0) { // Print 10 numbers per line
            printf("\n");
        }
    }
    printf("\n");
}

// Helper function to check if the array is sorted
int isSorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) return 0;
    }
    return 1;
}


int main(int argc, char *argv[]) {
    int N = 1024; // Default value for N
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    int num_blocks = (int)ceil((float)N / PER_BLOCK);

    printf("Testing with N = %d\n", N);

    struct timeval start_serial, end_serial, start_cuda, end_cuda;

    int *h_A = (int *)malloc(N * sizeof(int));
    int *m_A = (int *)malloc(N * sizeof(int));

    if (!h_A || !m_A) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        int random = rand() % N + 1;
        h_A[i] = random;
        m_A[i] = random;
    }

    // Print array before sorting
    printFullArray("Before Sorting", h_A, N);

    // Serial sorting
    gettimeofday(&start_serial, NULL);
    mergeSort(m_A, 0, N - 1);
    gettimeofday(&end_serial, NULL);

    // Print the array sorted by mergeSort
    // printFullArray("Array Sorted by mergeSort", m_A, N);

    size_t size = N * sizeof(int);
    int *d_A;
    cudaMalloc(&d_A, size);
    if (cudaMalloc(&d_A, size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_A failed!\n");
        free(h_A);
        free(m_A);
        return 1;
    }

    // Allocate temporary global memory for merge_chunks
    int *d_temp_global;
    cudaMalloc(&d_temp_global, size);
    if (cudaMalloc(&d_temp_global, size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for d_temp_global failed!\n");
        cudaFree(d_A);
        free(h_A);
        free(m_A);
        return 1;
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(PER_BLOCK);
    dim3 num_of_blocks(num_blocks);

    // CUDA sorting
    gettimeofday(&start_cuda, NULL);
    sample_sort<<<num_of_blocks, threads_per_block>>>(d_A, N);
    cudaDeviceSynchronize();

    int chunk_size = PER_BLOCK; // Start with block-sized chunks

    while (chunk_size < N) {
        int num_chunks = (N + 2 * chunk_size - 1) / (2 * chunk_size);
        merge_chunks<<<num_chunks, threads_per_block>>>(d_A, N, chunk_size, d_temp_global);
        cudaDeviceSynchronize();
        chunk_size *= 2;
    }

    gettimeofday(&end_cuda, NULL);

    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    // Print the entire sorted array
    printFullArray("Globally Sorted Array", h_A, N);

    // Check if GPU sort is correct
    if (!isSorted(h_A, N)) {
        printf("GPU sorting failed!\n");
        cudaFree(d_A);
        cudaFree(d_temp_global);
        free(h_A);
        free(m_A);
        return 1;
    }

    cudaFree(d_A);
    cudaFree(d_temp_global);

    printf("\nSerial Sort Time: %ld microseconds\n",
           (end_serial.tv_sec - start_serial.tv_sec) * 1000000 +
               (end_serial.tv_usec - start_serial.tv_usec));

    printf("\nCUDA Sort Time: %ld microseconds\n",
           (end_cuda.tv_sec - start_cuda.tv_sec) * 1000000 +
               (end_cuda.tv_usec - start_cuda.tv_usec));

    printf("\nSpeedup (Serial Sort Time / CUDA Sort Time): %.2f\n",
        (float)((end_serial.tv_sec - start_serial.tv_sec) * 1000000 +
                (end_serial.tv_usec - start_serial.tv_usec)) /
        (float)((end_cuda.tv_sec - start_cuda.tv_sec) * 1000000 +
                (end_cuda.tv_usec - start_cuda.tv_usec)));

    free(h_A);
    free(m_A);

    return 0;
}
