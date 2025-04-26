#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <vector>
#include "VectorOps.h"

namespace activations {
    class Sigmoid{
    private:

    public:
        std::vector<std::vector<double>> activation(const std::vector<std::vector<double>>& Z);
        std::vector<std::vector<double>> derivation(const std::vector<std::vector<double>>& Z);
    };

    class ReLU{
    private:

    public:
        std::vector<std::vector<double>> activation(const std::vector<std::vector<double>>& Z);
        std::vector<std::vector<double>> derivation(const std::vector<std::vector<double>>& Z);
    };

    class LeakyReLU{
    private:
        double negative_slope;

    public:
        LeakyReLU(double neg_slope=0.01);
        std::vector<std::vector<double>> activation(const std::vector<std::vector<double>>& Z);
        std::vector<std::vector<double>> derivation(const std::vector<std::vector<double>>& Z);
    };

    class tanh{
    private:

    public:
        std::vector<std::vector<double>> activation(const std::vector<std::vector<double>>& Z);
        std::vector<std::vector<double>> derivation(const std::vector<std::vector<double>>& Z);
    };

    class ELU{
    private:
        double alpha;

    public:
        ELU(double _alpha);
        std::vector<std::vector<double>> activation(const std::vector<std::vector<double>>& Z);
        std::vector<std::vector<double>> derivation(const std::vector<std::vector<double>>& Z);
    };

    class Swish{
    private:
        double beta;

    public:
        Swish(double _beta);
        std::vector<std::vector<double>> activation(const std::vector<std::vector<double>>& Z);
        std::vector<std::vector<double>> derivation(const std::vector<std::vector<double>>& Z);
    };
}

#endif // ACTIVATIONS_H