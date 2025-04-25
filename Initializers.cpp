#include <vector>
#include <random>
#include "Initializers.h"


// RandomInitializer class
RandomInitializer::RandomInitializer(const double& parameter_initialization_scale=0.01):
    param_init_scale(parameter_initialization_scale),
    rng(rd()),
    distribution(0, parameter_initialization_scale){}

std::vector<std::vector<double>> RandomInitializer::initialize_weights(const size_t& num_neurons,const size_t& num_neurons_previous){
    std::vector<std::vector<double>> weights(num_neurons, std::vector<double>(num_neurons_previous));
    for(int i = 0; i < num_neurons; i++){
        for(int j = 0; j < num_neurons_previous; j++){
            weights[i][j] = distribution(rng);
        }
    }
    return weights;
}

std::vector<double> RandomInitializer::initialize_biases(const int& num_neurons) {
    return std::vector<double>(num_neurons, 0);
}


// HeInitializer class
HeInitializer::HeInitializer():rng(rd()){}

std::vector<std::vector<double>> HeInitializer::initialize_weights(const size_t& num_neurons,const size_t& num_neurons_previous){
    double std = sqrt(2.0 / double(num_neurons_previous));
    std::vector<std::vector<double>> weights(num_neurons, std::vector<double>(num_neurons_previous));
    for(int i = 0; i < num_neurons; i++){
        for(int j = 0; j < num_neurons_previous; j++){
            weights[i][j] = distribution(rng);
        }
    }
    return weights;
}

std::vector<double> HeInitializer::initialize_biases(const size_t& num_neurons) {
    return std::vector<double>(num_neurons, 0);
}