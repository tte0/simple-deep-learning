#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace VectorOps {

    // Scalar addition                  DONE
    std::vector<std::vector<double>>    add(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise addition            DONE
    std::vector<std::vector<double>>    add(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Element-wise subtraction         DONE
    std::vector<std::vector<double>>    subtract(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Scalar multiplication            DONE
    std::vector<std::vector<double>>    multiply(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise multiplication      DONE
    std::vector<std::vector<double>>    multiply(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Element-wise division            DONE
    std::vector<std::vector<double>>    divide(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Element-wise square              DONE
    std::vector<std::vector<double>>    square(const std::vector<std::vector<double>>& a);

    // Element-wise sqrt                DONE
    std::vector<std::vector<double>>    sqrt(const std::vector<std::vector<double>>& a);

    // Element-wise exponential         DONE
    std::vector<std::vector<double>>    exp(const std::vector<std::vector<double>>& a);

    // Element-wise logarithm           DONE
    std::vector<std::vector<double>>    log(const std::vector<std::vector<double>>& a);

    // Element-wise sigmoid             DONE
    std::vector<std::vector<double>>    sigmoid(const std::vector<std::vector<double>>& a);

    // Element-wise tanh                DONE
    std::vector<std::vector<double>>    tanh(const std::vector<std::vector<double>>& a);

    // Scalar max                       DONE
    std::vector<std::vector<double>>    max(const std::vector<std::vector<double>>& a, double scalar);

    // Element-wise max                 DONE
    std::vector<std::vector<double>>    max(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);

    // Mean along axis=1                DONE
    std::vector<double>                 mean_axis1(const std::vector<std::vector<double>>& a);

    // Sum of all elements              DONE
    double                              sum(const std::vector<std::vector<double>>& a);

    // Sum of a vector                  DONE
    double                              sum(const std::vector<double>& a);

    // Sum along axis=0 with keepdims   DONE
    std::vector<std::vector<double>>    sum_axis0(const std::vector<std::vector<double>>& a);
}

#endif // VECTOR_OPS_H