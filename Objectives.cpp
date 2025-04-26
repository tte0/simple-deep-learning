#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "VectorOps.h"
#include "Objectives.h"

namespace objectives {
    // MeanSquaredError
    double MeanSquaredError::cost(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y){
        std::vector<std::vector<double>> diff = VectorOps::subtract(AL, Y);
        std::vector<std::vector<double>> squared_diff = VectorOps::square(diff);
        std::vector<double> mean_vals = VectorOps::mean_axis1(squared_diff);
        double sum_val = VectorOps::sum(mean_vals) / AL.size();
        return sum_val;
    }

    std::vector<std::vector<double>> MeanSquaredError::cost_prime(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y){
        size_t m = Y[0].size();
        std::vector<std::vector<double>> diff = VectorOps::subtract(Y, AL);
        std::vector<std::vector<double>> cost_derivative = VectorOps::multiply(diff, 2 * (1.0 / m));
        return cost_derivative;
    }


    // CrossEntropyLoss
    CrossEntropyLoss::CrossEntropyLoss(bool use_softmax=true) : use_softmax(use_softmax) {}

    std::vector<std::vector<double>> CrossEntropyLoss::softmax(const std::vector<std::vector<double>>& A) {
        std::vector<std::vector<double>> e_A = VectorOps::exp(A);
        std::vector<std::vector<double>> sum_e_A = VectorOps::sum_axis0(e_A);
        
        std::vector<std::vector<double>> result(A.size(), std::vector<double>(A[0].size()));
        for (size_t i = 0; i < A.size(); i++) {
            for (size_t j = 0; j < A[0].size(); j++) {
                result[i][j] = e_A[i][j] / sum_e_A[0][j];
            }
        }
        
        return result;
    }

    double CrossEntropyLoss::cost(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y) {
        std::vector<std::vector<double>> A = use_softmax ? softmax(AL) : AL;
        size_t m = Y[0].size();
        std::vector<std::vector<double>> log_A = VectorOps::log(A);
        std::vector<std::vector<double>> Y_log_A = VectorOps::multiply(Y, log_A);
        double cost = -VectorOps::sum(Y_log_A) / m;
        return cost;
    }

    std::vector<std::vector<double>> CrossEntropyLoss::cost_prime(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y) {
        size_t m = Y[0].size();

        return use_softmax ? VectorOps::subtract(AL, Y) :
                                VectorOps::multiply(VectorOps::divide(Y, VectorOps::multiply(AL, m)), -1);
    }
} // namespace objectives