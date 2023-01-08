#ifndef GP_CUDA_ADD_VECTORS_H
#define GP_CUDA_ADD_VECTORS_H

void cuda_add_vectors(float* a, float* b, float* c, int n);

// base cuda matrix multiplication
void cuda_matrix_multiplication(double* A, double* B, double* C, int rowsA, int colsA, int rowsB, int colsB);

// base cuda invert matrix
//void cuda_invert_matrix(double* A, double* Ainv, int N);
void cuda_matrix_inverse(double* A, int N);
void cuda_rbf_kernel(double* A, double* B, double* K, int r, int c, int feat, double gamma);
#endif  // GP_CUDA_ADD_VECTORS_H
