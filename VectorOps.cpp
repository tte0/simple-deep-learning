#include "VectorOps.h"
#include <cassert>
// Helper functions for vector operations
namespace VectorOps{

    // Scalar addition
    std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& a, double scalar){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = a[i][j] + scalar;
            }
        }
        return result;
    }

    // Element-wise addition
    std::vector<std::vector<double>> add(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b){
        assert(a.size() == b.size() && a[0].size() == b[0].size());
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    // Element-wise subtraction
    std::vector<std::vector<double>> subtract(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b){
        assert(a.size() == b.size() && a[0].size() == b[0].size());
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    // Scalar multiplication
    std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& a, double scalar){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }

    // Element-wise multiplication
    std::vector<std::vector<double>> multiply(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b){
        assert(a.size() == b.size() && a[0].size() == b[0].size());
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = a[i][j] * b[i][j];
            }
        }
        return result;
    }

    // Element-wise division
    std::vector<std::vector<double>> divide(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b){
        assert(a.size() == b.size() && a[0].size() == b[0].size());
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = a[i][j] / b[i][j];
            }
        }
        return result;
    }

    // Element-wise square
    std::vector<std::vector<double>> square(const std::vector<std::vector<double>>& a){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = a[i][j] * a[i][j];
            }
        }
        return result;
    }

    // Element-wise sqrt
    std::vector<std::vector<double>> sqrt(const std::vector<std::vector<double>>& a){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = std::sqrt(a[i][j]);
            }
        }
        return result;
    }

    // Element-wise exponential
    std::vector<std::vector<double>> exp(const std::vector<std::vector<double>>& a){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = std::exp(a[i][j]);
            }
        }
        return result;
    }

    // Element-wise logarithm
    std::vector<std::vector<double>> log(const std::vector<std::vector<double>>& a){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = std::log(a[i][j]);
            }
        }
        return result;
    }
    
    // Element-wise sigmoid
    std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>& a){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = 1.0 / (1.0 + std::exp(-a[i][j]));
            }
        }
        return result;
    }

    // Element-wise tanh
    std::vector<std::vector<double>> tanh(const std::vector<std::vector<double>>& a){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = std::tanh(a[i][j]);
            }
        }
        return result;
    }

    // Scalar max
    std::vector<std::vector<double>> max(const std::vector<std::vector<double>>& a, double scalar){
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = std::max(a[i][j], scalar);
            }
        }
        return result;
    }

    // Element-wise max
    std::vector<std::vector<double>> max(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b){
        assert(a.size() == b.size() && a[0].size() == b[0].size());
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j] = std::max(a[i][j], b[i][j]);
            }
        }
        return result;
    }

    // Sum of all elements
    double sum(const std::vector<std::vector<double>>& a) {
        double total = 0.0;
        for (const auto& row : a) {
            total += std::accumulate(row.begin(), row.end(), 0.0);
        }
        return total;
    }

    // Sum of a vector
    double sum(const std::vector<double>& a) {
        return std::accumulate(a.begin(), a.end(), 0.0);
    }

    // Mean along axis=1
    std::vector<double> mean_axis1(const std::vector<std::vector<double>>& a) {
        std::vector<double> result(a.size(), 0.0);
        for (size_t i = 0; i < a.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < a[0].size(); ++j) {
                sum += a[i][j];
            }
            result[i] = sum / a[0].size();
        }
        return result;
    }

    // Sum along axis=0 with keepdims
    std::vector<std::vector<double>> sum_axis0(const std::vector<std::vector<double>>& a) {
        std::vector<std::vector<double>> result(1, std::vector<double>(a[0].size(), 0.0));
        for (size_t j = 0; j < a[0].size(); ++j) {
            for (size_t i = 0; i < a.size(); ++i) {
                result[0][j] += a[i][j];
            }
        }
        return result;
    }
}