#ifndef OBJECTIVES_H
#define OBJECTIVES_H

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "VectorOps.h"

class MeanSquaredError{
private:

public:
    double cost(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y);
    std::vector<std::vector<double>> cost_prime(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y);
};

class CrossEntropyLoss{
private:
    bool use_softmax;

public:
    CrossEntropyLoss(bool use_softmax);
    std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& A);
    double cost(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y);
    std::vector<std::vector<double>> cost_prime(const std::vector<std::vector<double>>& AL, const std::vector<std::vector<double>>& Y);
};

#endif // OBJECTIVES_H