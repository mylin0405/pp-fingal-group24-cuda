#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void matrix_multiplication_kernel(double* A, double* B, double* C, int rowsA, int colsA, int colsB) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  //printf("lanunch kernel !!!");
  if (row < rowsA && col < colsB) {
    double sum = 0;
    for (int i = 0; i < colsA; i++) {
      sum += A[row * colsA + i] * B[i * colsB + col];
    }
    C[row * colsB + col] = sum;
  }
}

void cuda_matrix_multiplication(double* A, double* B, double* C, int rowsA, int colsA, int rowsB, int colsB) {
    int sizeA = rowsA * colsA;
    int sizeB = colsA * colsB;
    int sizeC = rowsA * colsB;
    double* d_A, *d_B, *d_C;

    // Allocate memory on the GPU for the matrices
    cudaMalloc((void**)&d_A, sizeA * sizeof(double));
    cudaMalloc((void**)&d_B, sizeB * sizeof(double));
    cudaMalloc((void**)&d_C, sizeC * sizeof(double));

    // Copy the matrices from the host to the GPU
    cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threads(32, 32);
    dim3 blocks((rowsA + threads.x - 1) / threads.x, (colsB + threads.y - 1) / threads.y);
    matrix_multiplication_kernel<<<blocks, threads>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    // Check for GPU errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
    return;
    }
    
    // Copy the result from the GPU to the host
    cudaMemcpy(C, d_C, sizeC * sizeof(double), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


__global__ void nodiag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == i && x!=y){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}
	
}

__global__ void diag_normalize(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(double *A, double *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}

}

__global__ void set_zero(double *A, double *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

void cuda_matrix_inverse(double* A, int n) {
	double *d_A, *I, *dI;
	int ddsize = n*n*sizeof(double);
	int blocksize = 32;
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);

	cudaMalloc((void**)&d_A, ddsize);
	cudaMalloc((void**)&dI, ddsize);

	I = new double[n*n];
	for (int i = 0; i<n; i++){
		for (int j = 0; j<n; j++){
			if (i == j) I[i*n + i] = 1.0;
			else I[i*n + j] = 0.0;
		}
	}
	
	cudaMemcpy(d_A, A, ddsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
	
	for (int i = 0; i<n; i++){
		nodiag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		diag_normalize << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		gaussjordan << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
		set_zero << <numBlocks, threadsPerBlock >> >(d_A, dI, n, i);
	}
	
	cudaMemcpy(A, dI, ddsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(dI);
}

__global__ void rbf(double* A, double* B, double* K, int r, int c, int feat, double gamma){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < r && y < c){
		double* A_row = A + x*feat;
		double* B_row = A + y*feat;
		double sum, dif;
		for(int i = 0;i < feat; ++i){
			dif = A_row[i] - B_row[i];
			sum += dif*dif;
		}
		K[x*c + y] = exp(-gamma*sum);
	}
}

void cuda_rbf_kernel(double* A, double* B, double* K, int r, int c, int feat, double gamma) {
	double *d_A, *d_B, *d_K;
	int K_size = r*c*sizeof(double);
	int A_size = r*feat*sizeof(double);
	int B_size = c*feat*sizeof(double);
	int blocksize = 32;
	dim3 threadsPerBlock(blocksize, blocksize);
	dim3 numBlocks((r + blocksize - 1) / blocksize, (c + blocksize - 1) / blocksize);

	cudaMalloc((void**)&d_A, A_size);
	cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
	if(A == B)
		d_B = d_A;
	else{
		cudaMalloc((void**)&d_B, B_size);
		cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice);
	}
	cudaMalloc((void**)&d_K, K_size);

	rbf<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_K, r, c, feat, gamma);
	
	cudaMemcpy(K, d_K, K_size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	if(A != B)
		cudaFree(d_B);
	cudaFree(d_K);
}
