#include <iostream>
#include <cuda_utils.hpp>
#include <gp_model_cuda.hpp>
#include <matrix/matrix.hpp>
#include <linalg/linalg.hpp>
#include <utils.hpp>
#include <gp/model.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

auto print_time_spent(std::chrono::high_resolution_clock::time_point start_time){
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = end_time - start_time;
    std::cout << "Time spent: " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
        << " (ms)"
        << std::endl;
    return end_time;
}

auto statistic(GP::matrix& m){
    auto&& [row, col] = m.shape();

    // mean
    GP::matrix mu{1, col};
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c)
            mu(0, c) += m(r, c);
    mu *= double{1./row};
    
    // var
    GP::matrix stdv{1, col};
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c){
            auto dif = (m(r, c) - mu(0, c));
            stdv(0, c) += dif*dif;
        }
    stdv *= double{1./row};
    for(size_t c = 0; c < col; ++ c){
        stdv(0, c) = sqrt(stdv(0, c));
    }
    return std::pair<GP::matrix, GP::matrix>{mu, stdv};
}


void preprocess(GP::matrix& m, const GP::matrix& mu, GP::matrix& stdv){
    auto&& [row, col] = m.shape();
    for(size_t r = 0; r < row; ++ r)
        for(size_t c = 0; c < col; ++ c)
            m(r, c) = stdv(0, c) != 0 ? ((m(r, c) - mu(0, c)) / stdv(0, c)) : (m(r, c) - mu(0, c));
}

void train_cuda(){
    using namespace GP::linalg;
    GP::matrix X, X_test;
    GP::matrix Y, Y_test;
    std::cin >> X >> Y;
    std::cin >> X_test >> Y_test;

    auto&& [col_mu_X, col_stdv_X] = statistic(X);
    preprocess(X, col_mu_X, col_stdv_X);
    preprocess(X_test, col_mu_X, col_stdv_X);

    std::cout << "train data size:" << X.shape().first << '\n';
    std::cout << "test data size:" << X_test.shape().first << '\n';
    int g = 1, b = 1;
    for(g = 1; g < 10001; g *= 10)
    for(b = 1; b < 10001; b *= 10) // grid search
    {
        std::cout << std::string(70, '-') << '\n';
        std::cout << "para: gamma = " << g*0.00003 <<", beta = " << b*0.01 << std::endl;
        GP::GPRegression_cuda model{g*0.000003, b*0.01};

        auto start = std::chrono::high_resolution_clock::now();
        model.fit(X, Y);
        std::cout << "[Fit] ";
        start = print_time_spent(start);
        auto&& [mu, var] = model.predict(X_test);
        std::cout << "[Predict] ";
        start = print_time_spent(start);

        // mse 
        auto diff = (mu - Y_test);
        std::cout << "MSE: " << (transpose(diff) ^ diff) * (1./diff.size())<<"\n";

        auto compare = GP::matrix{Y_test.size(), 2};
        for(size_t idx = 0; idx < Y_test.size(); ++idx){
            compare(idx, 0) = mu(idx, 0);
            compare(idx, 1) = Y_test(idx, 0);
        }
    }
}

void train(){
    using namespace GP::linalg;
    GP::matrix X, X_test;
    GP::matrix Y, Y_test;
    std::cin >> X >> Y;
    std::cin >> X_test >> Y_test;

    auto&& [col_mu_X, col_stdv_X] = statistic(X);
    preprocess(X, col_mu_X, col_stdv_X);
    preprocess(X_test, col_mu_X, col_stdv_X);

    std::cout << "train data size:" << X.shape().first << '\n';
    std::cout << "test data size:" << X_test.shape().first << '\n';
    int g = 1, b = 1;
    for(g = 1; g < 10001; g *= 10)
    for(b = 1; b < 10001; b *= 10) // grid search
    {
        std::cout << std::string(70, '-') << '\n';
        std::cout << "para: gamma = " << g*0.00003 <<", beta = " << b*0.01 << std::endl;
        GP::GPRegression model{g*0.000003, b*0.01};

        auto start = std::chrono::high_resolution_clock::now();
        model.fit(X, Y);
        std::cout << "[Fit] ";
        start = print_time_spent(start);
        auto&& [mu, var] = model.predict(X_test);
        std::cout << "[Predict] ";
        start = print_time_spent(start);

        // mse 
        auto diff = (mu - Y_test);
        std::cout << "MSE: " << (transpose(diff) ^ diff) * (1./diff.size())<<"\n";

        auto compare = GP::matrix{Y_test.size(), 2};
        for(size_t idx = 0; idx < Y_test.size(); ++idx){
            compare(idx, 0) = mu(idx, 0);
            compare(idx, 1) = Y_test(idx, 0);
        }
    }
}

void linalg_benchmark_serial(){
    using namespace GP::linalg;
    GP::matrix A{randn(100, 100)}, B{randn(300, 300)}, C{randn(1000, 1000)};
    int repeat = 1;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[serial benchmark start]..." << std::endl;
    std::cout << "[matmul] Repeat " << repeat <<" times \n";
    std::cout << "A 100x100 - ";
    for(int i = 0; i < repeat; ++i)
        A ^= A;
    start = print_time_spent(start);

    std::cout << "B 300x300 - ";
    for(int i = 0; i < repeat; ++i)
        B ^= B;
    start = print_time_spent(start);
    
    std::cout << "C 1000x1000 - ";
    for(int i = 0; i < repeat; ++i)
        C ^= C;
    start = print_time_spent(start);

    std::cout << "[inv] Repeat " << repeat <<" times \n";
        std::cout << "A 100x100 - ";
    for(int i = 0; i < repeat; ++i)
        A = ~A;
    start = print_time_spent(start);

    std::cout << "B 300x300 - ";
    for(int i = 0; i < repeat; ++i)
        B = ~B;
    start = print_time_spent(start);
    
    std::cout << "C 1000x1000 - ";
    for(int i = 0; i < repeat; ++i)
        C = ~C;
    start = print_time_spent(start);
    std::cout << "[serial test end] \n" << std::endl;
}

void linalg_benchmark_cuda(){
    using namespace GP::linalg;
    GP::matrix A{randn(100, 100)}, B{randn(300, 300)}, C{randn(1000, 1000)};
    GP::matrix A_backup = A;
    int repeat = 1;
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "[CUDA benchmark start]..." << std::endl;
    std::cout << "[Warming up multiplication] " << std::endl;
    auto A_ptr = A.ptr();
    cuda_matrix_multiplication(A_ptr, A_ptr, A_ptr, 100, 100, 100, 100);
    /* small validation here */
    std::cout << "CUDA Answer: " <<  A_ptr[0] << " " << A_ptr[1] << " " << A_ptr[203] <<std::endl;
    A_backup ^= A_backup;
    std::cout << "Serial Answer: " << A_backup.ptr()[0] << " " << A_backup.ptr()[1] << " " << A_backup.ptr()[203] << std::endl;
    start = print_time_spent(start);
    /* small validation here */
    std::cout << "[matmul] Repeat 5 times" << std::endl;
    std::cout << "A 100x100 - ";
    for (int i = 0; i < repeat; i++) {
        cuda_matrix_multiplication(A_ptr, A_ptr, A_ptr, 100, 100, 100, 100);
    }
    start = print_time_spent(start);
   
    std::cout << "B 300x300 - ";
    auto B_ptr = B.ptr();
    for (int i = 0; i < repeat; i++) {
        cuda_matrix_multiplication(B_ptr, B_ptr, B_ptr, 300, 300, 300, 300);
    }
    start = print_time_spent(start);
    //std::cout << B_ptr[0] << " " << B_ptr[1] << std::endl;

    std::cout << "C 1000x1000 - ";
    auto C_ptr = C.ptr();
    for (int i = 0; i < repeat; i++) {
        cuda_matrix_multiplication(C_ptr, C_ptr, C_ptr, 1000, 1000, 1000, 1000);
    }
    start = print_time_spent(start);
    //std::cout << C_ptr[0] << " " << C_ptr[1] << std::endl;
    std::cout << "[Warmning up inverse]" << std::endl;
    A_backup = A;
    cuda_matrix_inverse(A.ptr(), 100);
    A_backup = ~A_backup;
    std::cout << "Serial Answer: " << A_backup.ptr()[100] << " " << A_backup.ptr()[200] << " " << A_backup.ptr()[300] << std::endl;
    std::cout << "GPU Answer: " << A.ptr()[100] << " " << A.ptr()[200] << " " << A.ptr()[300] << std::endl; 
    start = print_time_spent(start);
    std::cout << "[inv] Repeat " << repeat <<" times \n";
    std::cout << "A 100x100 - ";
    for (int i = 0; i < repeat; i++) {
        cuda_matrix_inverse(A.ptr(), 100);
    }
    start = print_time_spent(start);

    std::cout << "B 300x300 - ";
    for(int i = 0; i < repeat; ++i) {
        cuda_matrix_inverse(B.ptr(), 300);
    }
    start = print_time_spent(start);

    std::cout << "C 1000x1000 - ";
    for(int i = 0; i < repeat; ++i) {
        cuda_matrix_inverse(C.ptr(), 1000);
    }
    start = print_time_spent(start);
    std::cout << "[cuda test end]" << std::endl;   
}

int main(int argc, const char* argv[]){
    using namespace GP::linalg;
    train_cuda();
    linalg_benchmark_serial();
    linalg_benchmark_cuda();
   /* 
    const int N = 4;
    double A[N * N] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };  
	cuda_invert_matrix(A, N);
	
    // Print the result
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", A[i * N + j]);
        }
        printf("\n");
    }
    */
    return 0;
}

