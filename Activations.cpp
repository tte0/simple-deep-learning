#include <math.h>
#include <vector>
#include "VectorOps.h"
using namespace std;

class Sigmoid{
private:

public:
    vector<vector<double>> activation(const vector<vector<double>>& Z){
        return VectorOps::sigmoid(Z);
    }

    vector<vector<double>> derivation(const vector<vector<double>>& Z){
        vector<vector<double>> sig = VectorOps::sigmoid(Z);
        vector<vector<double>> A(Z.size(), vector<double>(Z[0].size(), 1.0));// all elements 1.0
        return VectorOps::multiply(sig, VectorOps::subtract(A, sig));
    }
};

class ReLU{
private:

public:
    vector<vector<double>> activation(const vector<vector<double>>& Z){
        return VectorOps::max(Z, 0);
    }

    vector<vector<double>> derivation(const vector<vector<double>>& Z){
        vector<vector<double>> Z_prime(Z.size(), vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                Z_prime[i][j] = Z[i][j] > 0;
            }
        }
        return Z_prime;
    }
};

class LeakyReLU{
private:
    double negative_slope;

public:
    LeakyReLU(double neg_slope=0.01): negative_slope(neg_slope){}

    vector<vector<double>> activation(const vector<vector<double>>& Z){
        return VectorOps::max(Z, VectorOps::multiply(Z, negative_slope));
    }

    vector<vector<double>> derivation(const vector<vector<double>>& Z){
        vector<vector<double>> Z_prime(Z.size(), vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                Z_prime[i][j] = Z[i][j] > 0 ? 1 : negative_slope;
            }
        }
        return Z_prime;
    }
};

class tanh{
private:

public:
    vector<vector<double>> activation(const vector<vector<double>>& Z){
        return VectorOps::tanh(Z);
    }

    vector<vector<double>> derivation(const vector<vector<double>>& Z){
        vector<vector<double>> A(Z.size(), vector<double>(Z[0].size(), 1.0));// all elements 1.0
        vector<vector<double>> tanh_squared = VectorOps::square(VectorOps::tanh(Z));
        return VectorOps::subtract(A, tanh_squared);
    }
};

class ELU{
private:
    double alpha;

public:
    ELU(double _alpha=1.0):alpha(_alpha){}

    vector<vector<double>> activation(const vector<vector<double>>& Z){
        vector<vector<double>> A(Z.size(), vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                A[i][j] = Z[i][j] > 0 ? Z[i][j] : alpha * (std::exp(Z[i][j]) - 1);
            }
        }
        return A;
    }

    vector<vector<double>> derivation(const vector<vector<double>>& Z){
        vector<vector<double>> Z_prime(Z.size(), vector<double>(Z[0].size()));
        for(size_t i = 0; i < Z.size(); i++){
            for(size_t j = 0; j < Z[0].size(); j++){
                Z_prime[i][j] = Z[i][j] > 0 ? 1 : alpha * std::exp(Z[i][j]);
            }
        }
        return Z_prime;
    }
};

class Swish{
private:
    double beta;

public:
    Swish(double _beta=1.0):beta(_beta){}

    vector<vector<double>> activation(const vector<vector<double>>& Z){
        return VectorOps::multiply(Z, VectorOps::sigmoid(VectorOps::multiply(Z, beta)));
    }

    vector<vector<double>> derivation(const vector<vector<double>>& Z){
        vector<vector<double>> sigmoid_Z = VectorOps::sigmoid(VectorOps::multiply(Z, beta));
        vector<vector<double>> Z_prime = VectorOps::add(Z_prime, 1);
        Z_prime = VectorOps::multiply(sigmoid_Z, -1);
        Z_prime = VectorOps::multiply(Z_prime, VectorOps::multiply(Z, beta));
        Z_prime = VectorOps::add(Z_prime, 1);
        Z_prime = VectorOps::add(Z_prime, sigmoid_Z);
        return Z_prime;
    }
};