import numpy as np

class RandomInitializer:
    def __init__(self, parameter_initialization_scale=0.01):
        self.param_init_scale = parameter_initialization_scale

    def initialize_weights(self, num_neurons, num_neurons_previous):
        weights = np.random.randn(num_neurons, num_neurons_previous) * self.param_init_scale
        return weights

    def initialize_biases(self,  num_neurons):
        return np.zeros((num_neurons, 1))


class HeInitializer:
    def initialize_weights(self, num_neurons, num_neurons_previous):
        std = np.sqrt(2.0 / num_neurons_previous)
        weights = np.random.normal(0.0, std, (num_neurons, num_neurons_previous))
        return weights

    def initialize_biases(self, num_neurons):
        return np.zeros((num_neurons, 1))