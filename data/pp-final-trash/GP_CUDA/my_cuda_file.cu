#include <GP/linalg/linalg.hpp>
#include <matrix/matrix.hpp>
#include <random>
#include <unistd.h>
#include <cuda.h>

__global__ void add_vectors(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void cuda_add_vectors(float* a, float* b, float* c, int n) {
    using namespace GP::linalg;
    int num_threads = 256;
    int num_blocks = (n + num_threads - 1) / num_threads;
    int repeat = 5;
    int device_count;
    size_t free, total;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    cudaError_t total = cudaGetMemInfo(&free, &total);
    printf("Device Count is: %d \n", e);
    printf("Total Mem Size is: %d \n", total);
    printf("Total Mem Size is: %d \n", total);
    GP::matrix A{randn(100, 100)}, B{randn(300, 300)}, C{randn(1000, 1000)};
    for(int i = 0; i < repeat; ++i)
        A ^= A;
    add_vectors<<<num_blocks, num_threads>>>(a, b, c, n);
}