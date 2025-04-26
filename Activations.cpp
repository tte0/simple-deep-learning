#include <math.h>
#include <vector>
#include "VectorOps.h"
#include "Activations.h"

namespace activations {
    // Sigmoid 
    std::vector<std::vector<double>> Sigmoid::activation(const std::vector<std::vector<double>>& Z){
        return VectorOps::sigmoid(Z);
    }

    std::vector<std::vector<double>> Sigmoid::derivation(const std::vector<std::vector<double>>& Z){
        std::vector<std::vector<double>> sig = VectorOps::sigmoid(Z);
        std::vector<std::vector<double>> A(Z.size(), std::vector<double>(Z[0].size(), 1.0));// all elements 1.0
        return VectorOps::multiply(sig, VectorOps::subtract(A, sig));
    }


    // ReLU
    std::vector<std::vector<double>> ReLU::activation(const std::vector<std::vector<double>>& Z){
        return VectorOps::max(Z, 0);
    }

    std::vector<std::vector<double>> ReLU::derivation(const std::vector<std::vector<double>>& Z){
        std::vector<std::vector<double>> Z_prime(Z.size(), std::vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                Z_prime[i][j] = Z[i][j] > 0;
            }
        }
        return Z_prime;
    }


    // LeakyReLU
    LeakyReLU::LeakyReLU(double neg_slope=0.01): negative_slope(neg_slope){}

    std::vector<std::vector<double>> LeakyReLU::activation(const std::vector<std::vector<double>>& Z){
        return VectorOps::max(Z, VectorOps::multiply(Z, negative_slope));
    }

    std::vector<std::vector<double>> LeakyReLU::derivation(const std::vector<std::vector<double>>& Z){
        std::vector<std::vector<double>> Z_prime(Z.size(), std::vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                Z_prime[i][j] = Z[i][j] > 0 ? 1 : negative_slope;
            }
        }
        return Z_prime;
    }


    // tanh
    std::vector<std::vector<double>> tanh::activation(const std::vector<std::vector<double>>& Z){
        return VectorOps::tanh(Z);
    }

    std::vector<std::vector<double>> tanh::derivation(const std::vector<std::vector<double>>& Z){
        std::vector<std::vector<double>> A(Z.size(), std::vector<double>(Z[0].size(), 1.0));// all elements 1.0
        std::vector<std::vector<double>> tanh_squared = VectorOps::square(VectorOps::tanh(Z));
        return VectorOps::subtract(A, tanh_squared);
    }


    // ELU
    ELU::ELU(double _alpha=1.0):alpha(_alpha){}

    std::vector<std::vector<double>> ELU::activation(const std::vector<std::vector<double>>& Z){
        std::vector<std::vector<double>> A(Z.size(), std::vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                A[i][j] = Z[i][j] > 0 ? Z[i][j] : alpha * (std::exp(Z[i][j]) - 1);
            }
        }
        return A;
    }

    std::vector<std::vector<double>> ELU::derivation(const std::vector<std::vector<double>>& Z){
        std::vector<std::vector<double>> Z_prime(Z.size(), std::vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                Z_prime[i][j] = Z[i][j] > 0 ? 1 : alpha * std::exp(Z[i][j]);
            }
        }
        return Z_prime;
    }


    // Swish
    Swish::Swish(double _beta=1.0):beta(_beta){}

    std::vector<std::vector<double>> Swish::activation(const std::vector<std::vector<double>>& Z){
        return VectorOps::multiply(Z, VectorOps::sigmoid(VectorOps::multiply(Z, beta)));
    }

    std::vector<std::vector<double>> Swish::derivation(const std::vector<std::vector<double>>& Z){
        std::vector<std::vector<double>> sigmoid_Z = VectorOps::sigmoid(VectorOps::multiply(Z, beta));
        std::vector<std::vector<double>> Z_prime = VectorOps::add(Z_prime, 1);
        Z_prime = VectorOps::multiply(sigmoid_Z, -1);
        Z_prime = VectorOps::multiply(Z_prime, VectorOps::multiply(Z, beta));
        Z_prime = VectorOps::add(Z_prime, 1);
        Z_prime = VectorOps::add(Z_prime, sigmoid_Z);
        return Z_prime;
    }
} // namespace activations