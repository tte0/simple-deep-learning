#include "VectorOps.h"
using namespace std;

// Helper functions for vector operations
namespace VectorOps {

    // Scalar addition
    vector<vector<double>> add(const vector<vector<double>>& a, double scalar){
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for(size_t i = 0; i < a.size(); i++){
            for(size_t j = 0; j < a[0].size(); j++){
                result[i][j]=a[i][j]+scalar;
            }
        }
        return result;
    }

    // Element-wise subtraction
    vector<vector<double>> subtract(const vector<vector<double>>& a, const vector<vector<double>>& b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    // Element-wise multiplication
    vector<vector<double>> multiply(const vector<vector<double>>& a, 
                                             const vector<vector<double>>& b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] * b[i][j];
            }
        }
        return result;
    }

    // Element-wise square
    vector<vector<double>> square(const vector<vector<double>>& a) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] * a[i][j];
            }
        }
        return result;
    }

    // Mean along axis=1
    vector<double> mean_axis1(const vector<vector<double>>& a) {
        vector<double> result(a.size(), 0.0);
        for (size_t i = 0; i < a.size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < a[0].size(); ++j) {
                sum += a[i][j];
            }
            result[i] = sum / a[0].size();
        }
        return result;
    }

    // Element-wise division
    vector<vector<double>> divide(const vector<vector<double>>& a, 
                                           const vector<vector<double>>& b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] / b[i][j];
            }
        }
        return result;
    }

    // Scalar division
    vector<vector<double>> divide(const vector<vector<double>>& a, double scalar) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] / scalar;
            }
        }
        return result;
    }

    // Sum of all elements
    double sum(const vector<vector<double>>& a) {
        double total = 0.0;
        for (const auto& row : a) {
            for (double val : row) {
                total += val;
            }
        }
        return total;
    }

    // Sum of a vector
    double sum(const vector<double>& a) {
        return accumulate(a.begin(), a.end(), 0.0);
    }

    // Element-wise logarithm
    vector<vector<double>> log(const vector<vector<double>>& a) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = std::log(a[i][j]);
            }
        }
        return result;
    }

    // Element-wise exponential
    vector<vector<double>> exp(const vector<vector<double>>& a) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = std::exp(a[i][j]);
            }
        }
        return result;
    }

    // Element-wise negative exponential
    vector<vector<double>> neg_exp(const vector<vector<double>>& a) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = std::exp(a[i][j]);
            }
        }
        return result;
    }

    // Sum along axis=0 with keepdims
    vector<vector<double>> sum_axis0(const vector<vector<double>>& a) {
        vector<vector<double>> result(1, vector<double>(a[0].size(), 0.0));
        for (size_t j = 0; j < a[0].size(); ++j) {
            for (size_t i = 0; i < a.size(); ++i) {
                result[0][j] += a[i][j];
            }
        }
        return result;
    }

    // Element-wise negative division
    std::vector<std::vector<double>> VectorOps::neg_divide(const std::vector<std::vector<double>>& a, 
                                                        const std::vector<std::vector<double>>& b) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = -a[i][j] / b[i][j];
            }
        }
        return result;
    }

    // Scalar multiplication
    std::vector<std::vector<double>> VectorOps::multiply(const std::vector<std::vector<double>>& a, double scalar) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }
}