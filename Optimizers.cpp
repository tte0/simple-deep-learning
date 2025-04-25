#include <assert.h>
#include <tuple>
#include <vector>
#include "VectorOps.h"
using namespace std;


class GradientDescent{
private:
    double learning_rate;

public:
    GradientDescent(double _learning_rate):learning_rate(_learning_rate){}

    void build(char _, char __){
        return;
    }

    tuple<vector<vector<double>>,vector<vector<double>>>
    update_parameters(vector<vector<double>> weights_gradient, vector<vector<double>> biases_gradient){
        return{
            VectorOps::multiply(weights_gradient, -learning_rate),
            VectorOps::multiply(biases_gradient, -learning_rate)
        };
    }
};

class MomentumOptimizer{
private:
    double learning_rate;
    double beta;
    bool _built = false;
    vector<vector<double>> moment_W;
    vector<vector<double>> moment_b;

public:
    MomentumOptimizer(double _learning_rate, double _beta):
        learning_rate(_learning_rate),
        beta(_beta){}

    void build(pair<size_t,size_t> weights_shape, pair<size_t,size_t> biases_shape){
        if(_built) return; // Avoid re-initialization
        moment_W.assign(weights_shape.first, vector<double>(weights_shape.second));
        moment_b.assign(biases_shape.first, vector<double>(biases_shape.second));
        _built = true;
    }

    tuple<vector<vector<double>>,vector<vector<double>>>
    update_parameters(vector<vector<double>> weights_gradient, vector<vector<double>> biases_gradient){
        assert(_built);
        moment_W = VectorOps::add(VectorOps::multiply(moment_W, beta), VectorOps::multiply(weights_gradient, 1 - beta));
        moment_b = VectorOps::add(VectorOps::multiply(moment_b, beta), VectorOps::multiply(biases_gradient, 1 - beta));
        return{
            VectorOps::multiply(weights_gradient, -learning_rate),
            VectorOps::multiply(biases_gradient, -learning_rate)
        };
    }
};

class AdamOptimizer{
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    size_t timestep = 0;
    bool _built = false;
    vector<vector<double>> moment1_W;
    vector<vector<double>> moment1_b;
    vector<vector<double>> moment2_W;
    vector<vector<double>> moment2_b;
    double p_beta1 = 1;
    double p_beta2 = 1;

public:
    AdamOptimizer(double _learning_rate, double _beta1, double _beta2, double _epsilon):
        learning_rate(_learning_rate),
        beta1(_beta1),
        beta2(_beta2),
        epsilon(_epsilon){}

    void build(pair<size_t,size_t> weights_shape, pair<size_t,size_t> biases_shape){
        if(_built) return; // Avoid re-initialization
        moment1_W.assign(weights_shape.first, vector<double>(weights_shape.second));
        moment2_W.assign(weights_shape.first, vector<double>(weights_shape.second));
        moment1_b.assign(biases_shape.first, vector<double>(biases_shape.second));
        moment2_b.assign(biases_shape.first, vector<double>(biases_shape.second));
        _built = true;
    }

    tuple<vector<vector<double>>,vector<vector<double>>>
    update_parameters(vector<vector<double>> weights_gradient, vector<vector<double>> biases_gradient){
        assert(_built);
        timestep++;
        p_beta1 *= beta1;
        p_beta2 *= beta2;
        
        moment1_W = VectorOps::add(VectorOps::multiply(moment1_W, beta1), VectorOps::multiply(weights_gradient, 1 - beta1));
        moment1_b = VectorOps::add(VectorOps::multiply(moment1_b, beta1), VectorOps::multiply(biases_gradient, 1 - beta1));
        
        moment2_W = VectorOps::add(VectorOps::multiply(moment2_W, beta2), VectorOps::multiply(VectorOps::square(weights_gradient), 1 - beta2));
        moment2_b = VectorOps::add(VectorOps::multiply(moment2_b, beta2), VectorOps::multiply(VectorOps::square(biases_gradient), 1 - beta2));
        
        vector<vector<double>> corrected_moment1_W = VectorOps::divide(moment1_W, p_beta1);
        vector<vector<double>> corrected_moment1_b = VectorOps::divide(moment1_b, p_beta1);
    
        vector<vector<double>> corrected_moment2_W = VectorOps::divide(moment2_W, p_beta2);
        vector<vector<double>> corrected_moment2_b = VectorOps::divide(moment2_b, p_beta2);

        return{
            VectorOps::multiply(VectorOps::divide(corrected_moment1_W, VectorOps::add(VectorOps::sqrt(corrected_moment2_W), epsilon)), -learning_rate),
            VectorOps::multiply(VectorOps::divide(corrected_moment1_b, VectorOps::add(VectorOps::sqrt(corrected_moment2_b), epsilon)), -learning_rate),
        };
    }
};