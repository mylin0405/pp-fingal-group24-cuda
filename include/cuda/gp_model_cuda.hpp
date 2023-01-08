#pragma once
#include <matrix/matrix.hpp>
#include <linalg/linalg.hpp>
#include <cuda_utils.hpp>
#include <cmath>
#include <numeric>
#include <vector>
#include <iostream>
#include <chrono>
#include <stdio.h>
namespace GP{

class GPRegression_cuda{
private:
    matrix C_inv_, X_, Y_;
    double gamma_;
    double beta_;
    matrix rbf_kernel_(const matrix&, const matrix&);
public:
    GPRegression_cuda(double g = 0.1, double b = double{1}):
        C_inv_{}, X_{}, Y_{}, gamma_{g}, beta_{b} {}
	auto print_time_spent(std::chrono::high_resolution_clock::time_point start_time){
    	auto end_time = std::chrono::high_resolution_clock::now();
    	auto duration = end_time - start_time;
    	std::cout << "Time spent: "
        	<< std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
        	<< " (ms)"
        	<< std::endl;
    return end_time;
	}                            
    void fit(const matrix& X, const matrix& Y){
        using namespace GP::linalg;
        auto&& [xr, xc] = X.shape();
        auto&& [yr, yc] = Y.shape();
        if(xr != yr){
            throw matrix::DimensionalityException();
        }
        X_ = X; Y_ = Y;
        auto start_time = std::chrono::high_resolution_clock::now();
        // C_inv_ = rbf_kernel_(X, X) + identity(xr) * beta_;
        C_inv_ = matrix(xr);
        cuda_rbf_kernel(X_.ptr(), X_.ptr(), C_inv_.ptr(), xr, xr, xc, gamma_);
        C_inv_ = C_inv_ + identity(xr) * beta_;
        std::cout << "rbf_kernel time: ";
		start_time = print_time_spent(start_time);
        //std::cout << "rbf_kernel time: " << end_time - start_time << std::endl;
        auto&& [C_inv_r, C_inv_c] = C_inv_.shape();
        //std::cout << C_inv_r << " " << C_inv_c << std::endl;
        cuda_matrix_inverse(C_inv_.ptr(), C_inv_r);
        std::cout << "inverse time: ";
		start_time = print_time_spent(start_time);
        //std::cout << "Inverse time: " << end_time - start_time << std::endl;
    }
    auto predict(const matrix& X_test){
        using namespace GP::linalg;
        auto&& [xr, xc] = X_.shape();
        auto&& [xtest_r, xtest_c] = X_test.shape();
        if(xc != xtest_c){
            throw matrix::DimensionalityException();
        }
        // auto k = rbf_kernel_(X_, X_test);
        auto k = matrix(xr, xtest_r);
        cuda_rbf_kernel(X_.ptr(), X_test.ptr(), k.ptr(), xr, xtest_r, xc, gamma_);
        //auto ktCinv = transpose(k) ^ C_inv_;
		auto&& [C_inv_r, C_inv_c] = C_inv_.shape();
        auto k_transpose = transpose(k); 
		auto&& [kr, kc] = k_transpose.shape();
        auto ktCinv = matrix(kr, C_inv_c);
		cuda_matrix_multiplication(k_transpose.ptr(), C_inv_.ptr(), ktCinv.ptr(), kr, kc, C_inv_r, C_inv_c);
        auto&& [yr, yc] = Y_.shape();
        auto ktCinv_Y = matrix(kr, yc); 
        cuda_matrix_multiplication(ktCinv.ptr(), Y_.ptr(), ktCinv_Y.ptr(), kr, C_inv_c, yr, yc); // ktCinv ^ Y_;
        auto ktCinv_k = matrix(kr, kr);
        cuda_matrix_multiplication(ktCinv.ptr(), k.ptr(), ktCinv_k.ptr(), kr, C_inv_c, kc, kr); // ktCinv ^ k;
        auto xtest_k = matrix(xtest_r);
        cuda_rbf_kernel(X_test.ptr(), X_test.ptr(), xtest_k.ptr(), xtest_r, xtest_r, xc, gamma_);
        return std::pair<matrix, matrix>{
            ktCinv_Y,
            xtest_k + identity(xtest_r) * beta_ - ktCinv_k
        };
    }
};

matrix GPRegression_cuda::rbf_kernel_(const matrix& X1, const matrix& X2){
    using namespace GP::linalg;
    auto&& [n1, feats1] = X1.shape();
    auto&& [n2, feats2] = X2.shape();
    if(feats1 != feats2){
        throw matrix::DimensionalityException();
    }
    matrix kernel(n1, n2);
    for(size_t r = 0; r < n1; ++r){
        for(size_t c = 0; c < n2; ++c){
            std::vector<double> vec_dif(feats1);
            for(size_t k = 0; k < feats1; ++k)
                vec_dif[k] = X1(r, k) - X2(c, k);
            double dot_product = std::inner_product(
                vec_dif.begin(), vec_dif.end(), vec_dif.begin(), double{});
            kernel(r, c) = std::exp(-gamma_*dot_product);
        }
    }
    return kernel;
}

}
