#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <assert.h>
#include <tuple>
#include <vector>
#include "VectorOps.h"

namespace optimizers {
    class GradientDescent{
    private:
        double learning_rate;

    public:
        GradientDescent(double _learning_rate);
        void build(std::pair<size_t,size_t> weights_shape, std::pair<size_t,size_t> biases_shape);
        std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<double>>> 
        update_parameters(std::vector<std::vector<double>> weights_gradient, std::vector<std::vector<double>> biases_gradient);
    };

    class MomentumOptimizer{
    private:
        double learning_rate;
        double beta;
        bool _built = false;
        std::vector<std::vector<double>> moment_W;
        std::vector<std::vector<double>> moment_b;

    public:
        MomentumOptimizer(double _learning_rate, double _beta);
        void build(std::pair<size_t,size_t> weights_shape, std::pair<size_t,size_t> biases_shape);
        std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<double>>>
        update_parameters(std::vector<std::vector<double>> weights_gradient, std::vector<std::vector<double>> biases_gradient);
    };

    class AdamOptimizer{
    private:
        double learning_rate;
        double beta1;
        double beta2;
        double epsilon;
        size_t timestep = 0;
        bool _built = false;
        std::vector<std::vector<double>> moment1_W;
        std::vector<std::vector<double>> moment1_b;
        std::vector<std::vector<double>> moment2_W;
        std::vector<std::vector<double>> moment2_b;
        double p_beta1 = 1;
        double p_beta2 = 1;

    public:
        AdamOptimizer(double _learning_rate, double _beta1, double _beta2, double _epsilon);
        void build(std::pair<size_t,size_t> weights_shape, std::pair<size_t,size_t> biases_shape);
        std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<double>>>
        update_parameters(std::vector<std::vector<double>> weights_gradient, std::vector<std::vector<double>> biases_gradient);
    };
} // namespace optimizers


#endif // OPTIMIZER_H