#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include <vector>
#include <random>

class RandomInitializer{
private:
    double param_init_scale;
    std::random_device rd;
    std::mt19937 rng;
    std::normal_distribution<double> distribution;

public:
    RandomInitializer(const double& parameter_initialization_scale=0.01);
    std::vector<std::vector<double>> initialize_weights(const size_t& num_neurons,const size_t& num_neurons_previous);
    std::vector<double> initialize_biases(const int& num_neurons);
};

class HeInitializer{
private:
    std::random_device rd;
    std::mt19937 rng;
    std::normal_distribution<double> distribution;

public:
    HeInitializer();
    std::vector<std::vector<double>> initialize_weights(const size_t& num_neurons,const size_t& num_neurons_previous);
    std::vector<double> initialize_biases(const size_t& num_neurons);
};

#endif // INITIALIZERS_H