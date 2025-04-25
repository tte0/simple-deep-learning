#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "VectorOps.h"
using namespace std;

class MeanSquaredError{
private:

public:
    double cost(const vector<vector<double>>& AL, const vector<vector<double>>& Y){
        vector<vector<double>> diff = VectorOps::subtract(AL, Y);
        vector<vector<double>> squared_diff = VectorOps::square(diff);
        vector<double> mean_vals = VectorOps::mean_axis1(squared_diff);
        double sum_val = VectorOps::sum(mean_vals) / AL.size();
        return sum_val;
    }

    vector<vector<double>> cost_prime(const vector<vector<double>>& AL, const vector<vector<double>>& Y){
        size_t m = Y[0].size();
        vector<vector<double>> diff = VectorOps::subtract(Y, AL);
        vector<vector<double>> cost_derivative = VectorOps::multiply(diff, 2 * (1.0 / m));
        return cost_derivative;
    }
};

class CrossEntropyLoss{
private:
    bool use_softmax;

public:
    CrossEntropyLoss(bool use_softmax=true) : use_softmax(use_softmax) {}

    vector<vector<double>> softmax(const vector<vector<double>>& A) {
        vector<vector<double>> e_A = VectorOps::exp(A);
        vector<vector<double>> sum_e_A = VectorOps::sum_axis0(e_A);
        
        vector<vector<double>> result(A.size(), vector<double>(A[0].size()));
        for (size_t i = 0; i < A.size(); i++) {
            for (size_t j = 0; j < A[0].size(); j++) {
                result[i][j] = e_A[i][j] / sum_e_A[0][j];
            }
        }
        
        return result;
    }

    double cost(const vector<vector<double>>& AL, const vector<vector<double>>& Y) {
        vector<vector<double>> A = use_softmax ? softmax(AL) : AL;
        size_t m = Y[0].size();
        vector<vector<double>> log_A = VectorOps::log(A);
        vector<vector<double>> Y_log_A = VectorOps::multiply(Y, log_A);
        double cost = -VectorOps::sum(Y_log_A) / m;
        return cost;
    }

    vector<vector<double>> cost_prime(const vector<vector<double>>& AL, const vector<vector<double>>& Y) {
        size_t m = Y[0].size();
    
        return use_softmax ? VectorOps::subtract(AL, Y) :
                             VectorOps::multiply(VectorOps::divide(Y, VectorOps::multiply(AL, m)), -1);
    }
};