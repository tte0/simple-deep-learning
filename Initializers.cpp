#include <vector>
#include <random>
using namespace std;

class RandomInitializer{
private:
    double param_init_scale;
    random_device rd;
    mt19937 rng;
    normal_distribution<double> distribution;

public:
    RandomInitializer(double parameter_initialization_scale=0.01):
        param_init_scale(parameter_initialization_scale),
        rng(rd()),
        distribution(0, parameter_initialization_scale) {}

    vector<vector<double>> initialize_weights(int num_neurons,int num_neurons_previous){
        vector<vector<double>> weights(num_neurons, vector<double>(num_neurons_previous));
        for(int i = 0; i < num_neurons; i++){
            for(int j = 0; j < num_neurons_previous; j++){
                weights[i][j] = distribution(rng);
            }
        }
        return weights;
    }

    vector<double> initialize_biases(int num_neurons) {
        return vector<double>(num_neurons, 0);
    }

};

class HeInitializer{
private:
    std::random_device rd;
    std::mt19937 rng;
    normal_distribution<double> distribution;

public:
    HeInitializer() : rng(rd()) {}

    vector<vector<double>> initialize_weights(int num_neurons,int num_neurons_previous){
        double std = sqrt(2.0 / double(num_neurons_previous));
        vector<vector<double>> weights(num_neurons, vector<double>(num_neurons_previous));
        for(int i = 0; i < num_neurons; i++){
            for(int j = 0; j < num_neurons_previous; j++){
                weights[i][j] = distribution(rng);
            }
        }
        return weights;
    }

    vector<double> initialize_biases(int num_neurons) {
        return vector<double>(num_neurons, 0);
    }
};