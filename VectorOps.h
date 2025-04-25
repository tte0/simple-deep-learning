#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace VectorOps {

    // Scalar addition
    std::vector<std::vector<double>>    add(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise addition
    std::vector<std::vector<double>>    add(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Scalar subtraction
    std::vector<std::vector<double>>    subtract(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise subtraction
    std::vector<std::vector<double>>    subtract(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Scalar multiplication
    std::vector<std::vector<double>>    multiply(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise multiplication
    std::vector<std::vector<double>>    multiply(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Scalar division
    std::vector<std::vector<double>>    divide(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise division
    std::vector<std::vector<double>>    divide(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Element-wise square
    std::vector<std::vector<double>>    square(const std::vector<std::vector<double>>& a);

    // Element-wise sqrt
    std::vector<std::vector<double>>    sqrt(const std::vector<std::vector<double>>& a);

    // Element-wise exponential
    std::vector<std::vector<double>>    exp(const std::vector<std::vector<double>>& a);

    // Element-wise logarithm
    std::vector<std::vector<double>>    log(const std::vector<std::vector<double>>& a);

    // Element-wise sigmoid
    std::vector<std::vector<double>>    sigmoid(const std::vector<std::vector<double>>& a);

    // Element-wise tanh
    std::vector<std::vector<double>>    tanh(const std::vector<std::vector<double>>& a);

    // Scalar max
    std::vector<std::vector<double>>    max(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise max
    std::vector<std::vector<double>>    max(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Mean along axis=1
    std::vector<double>                 mean_axis1(const std::vector<std::vector<double>>& a);

    // Sum of all elements
    double                              sum(const std::vector<std::vector<double>>& a);

    // Sum of a vector
    double                              sum(const std::vector<double>& a);

    // Sum along axis=0 with keepdims
    std::vector<std::vector<double>>    sum_axis0(const std::vector<std::vector<double>>& a);
}

#endif // VECTOR_OPS_H