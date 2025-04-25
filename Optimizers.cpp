#include <assert.h>
#include <tuple>
#include <vector>
#include "VectorOps.h"
#include "Optimizers.h"


// GradientDescent
GradientDescent::GradientDescent(double _learning_rate):learning_rate(_learning_rate){}

void GradientDescent::build(std::pair<size_t,size_t> weights_shape, std::pair<size_t,size_t> biases_shape){
    // No initialization needed for GradientDescent
}

std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<double>>>
GradientDescent::update_parameters(std::vector<std::vector<double>> weights_gradient, std::vector<std::vector<double>> biases_gradient){
    return{
        VectorOps::multiply(weights_gradient, -learning_rate),
        VectorOps::multiply(biases_gradient, -learning_rate)
    };
}


// MomentumOptimizer
MomentumOptimizer::MomentumOptimizer(double _learning_rate, double _beta):
    learning_rate(_learning_rate),
    beta(_beta){}

void MomentumOptimizer::build(std::pair<size_t,size_t> weights_shape, std::pair<size_t,size_t> biases_shape){
    if(_built) return; // Avoid re-initialization
    moment_W.assign(weights_shape.first, std::vector<double>(weights_shape.second));
    moment_b.assign(biases_shape.first, std::vector<double>(biases_shape.second));
    _built = true;
}

std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<double>>>
MomentumOptimizer::update_parameters(std::vector<std::vector<double>> weights_gradient, std::vector<std::vector<double>> biases_gradient){
    assert(_built);
    moment_W = VectorOps::add(VectorOps::multiply(moment_W, beta), VectorOps::multiply(weights_gradient, 1 - beta));
    moment_b = VectorOps::add(VectorOps::multiply(moment_b, beta), VectorOps::multiply(biases_gradient, 1 - beta));
    return{
        VectorOps::multiply(weights_gradient, -learning_rate),
        VectorOps::multiply(biases_gradient, -learning_rate)
    };
}


// AdamOptimizer
AdamOptimizer::AdamOptimizer(double _learning_rate, double _beta1, double _beta2, double _epsilon):
    learning_rate(_learning_rate),
    beta1(_beta1),
    beta2(_beta2),
    epsilon(_epsilon){}

void AdamOptimizer::build(std::pair<size_t,size_t> weights_shape, std::pair<size_t,size_t> biases_shape){
    if(_built) return; // Avoid re-initialization
    moment1_W.assign(weights_shape.first, std::vector<double>(weights_shape.second));
    moment2_W.assign(weights_shape.first, std::vector<double>(weights_shape.second));
    moment1_b.assign(biases_shape.first, std::vector<double>(biases_shape.second));
    moment2_b.assign(biases_shape.first, std::vector<double>(biases_shape.second));
    _built = true;
}

std::tuple<std::vector<std::vector<double>>,std::vector<std::vector<double>>>
AdamOptimizer::update_parameters(std::vector<std::vector<double>> weights_gradient, std::vector<std::vector<double>> biases_gradient){
    assert(_built);
    timestep++;
    p_beta1 /= beta1;
    p_beta2 /= beta2;
    
    moment1_W = VectorOps::add(VectorOps::multiply(moment1_W, beta1), VectorOps::multiply(weights_gradient, 1 - beta1));
    moment1_b = VectorOps::add(VectorOps::multiply(moment1_b, beta1), VectorOps::multiply(biases_gradient, 1 - beta1));
    
    moment2_W = VectorOps::add(VectorOps::multiply(moment2_W, beta2), VectorOps::multiply(VectorOps::square(weights_gradient), 1 - beta2));
    moment2_b = VectorOps::add(VectorOps::multiply(moment2_b, beta2), VectorOps::multiply(VectorOps::square(biases_gradient), 1 - beta2));
    
    std::vector<std::vector<double>> corrected_moment1_W = VectorOps::multiply(moment1_W, p_beta1);
    std::vector<std::vector<double>> corrected_moment1_b = VectorOps::multiply(moment1_b, p_beta1);

    std::vector<std::vector<double>> corrected_moment2_W = VectorOps::multiply(moment2_W, p_beta2);
    std::vector<std::vector<double>> corrected_moment2_b = VectorOps::multiply(moment2_b, p_beta2);

    return{
        VectorOps::multiply(VectorOps::divide(corrected_moment1_W, VectorOps::add(VectorOps::sqrt(corrected_moment2_W), epsilon)), -learning_rate),
        VectorOps::multiply(VectorOps::divide(corrected_moment1_b, VectorOps::add(VectorOps::sqrt(corrected_moment2_b), epsilon)), -learning_rate),
    };
}
