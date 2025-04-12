import numpy as np
import copy


#
# PARAMETER INITIALIZATION
#

def initialize_parameters_deep(layers_dims, param_init_scale):
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * param_init_scale
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def initialize_velocity(parameters):
    L = len(parameters) // 2
    velocities = {}
    for l in range(L):
        velocities["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        velocities["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return velocities


#
# LEARNING FUNCTIONS
#


def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def update_parameters_velocity(params, grads, velocity, learning_rate, momentum_factor):
    parameters = copy.deepcopy(params)
    velocities = copy.deepcopy(velocity)
    L = len(parameters) // 2

    for l in range(L):
        velocities["dW" + str(l + 1)] = momentum_factor * velocities["dW" + str(l + 1)] + (1 - momentum_factor) * grads["dW" + str(l + 1)]
        velocities["db" + str(l + 1)] = momentum_factor * velocities["db" + str(l + 1)] + (1 - momentum_factor) * grads["db" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * velocities["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * velocities["db" + str(l + 1)]

    return parameters, velocities

#
# FORWARD PROPAGATION
#

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = np.dot(W, A_prev) + b
        A = sigmoid(Z)
    elif activation == "relu":
        Z = (np.dot(W, A_prev)) + b
        A = relu(Z)
    else:
        Z, A = -1
    cache = (A_prev, W, b, Z)
    return A, cache


def L_model_forward(X, parameters, activations):
    caches = []
    A = X # Set A to X before A_prev
    L = len(parameters) // 2 # Fixed mistake, layer_dims can't be accessed in function, so integer divide is used
    for l in range(1, L+1):
        A_prev = A # set A_prev to A at the beginning of the loop
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, activations[l-1])
        caches.append(cache)
    return A, caches


#
# BACKWARD PROPAGATION
#


def linear_activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    m = A_prev.shape[1]
    if activation == "sigmoid":
        dZ = dA * sigmoid_prime(Z)
    elif activation == "relu":
        dZ = dA * relu_prime(Z)
    else:
        dZ = -1
    dA_prev = np.dot(W.T, dZ) #  dA_prev.shape = (n[l-1], m) dZ.shape = (n[l], m) W.shape = (n[l], n[l-1])
    dW = np.dot(dZ, A_prev.T) # A_prev.shape = (n[l-1], m) dZ.shape = (n[l], m) dW.shape = (n[l], n[l-1])
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dA_prev, dW, db


def L_model_backward(dAL, caches, activations):
    grads = {}
    L = len(caches)
    dA_prev = dAL
    #m = dAL.shape[1]
    #Y = Y.reshape(dAL.shape)
    for l in reversed(range(L)):
        dA_prev, dW, db = linear_activation_backward(dA_prev, caches[l], activations[l-1])
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    return grads


#
# COST FUNCTIONS
#


def compute_cost(AL, Y, loss_function):
    if loss_function == "mean_squared_error":
        cost = mean_squared_error(AL, Y)
    elif loss_function == "cross_entropy_loss":
        cost = cross_entropy_loss(AL, Y)
    else:
        return -1
    return cost


def compute_cost_derivative(AL, Y, loss_function):
    if loss_function == "mean_squared_error":
        cost_derivative = mean_squared_error_prime(AL, Y)
    elif loss_function == "cross_entropy_loss":
        cost_derivative = cross_entropy_loss_prime(AL, Y)
    else:
        return -1
    return cost_derivative


def mean_squared_error(AL, Y):
    m = Y.shape[1]
    cost = np.squeeze(np.sum(np.mean(np.square(AL - Y), axis=1, keepdims=False))) / AL.shape[0]
    cost = np.squeeze(cost)  # squeeze any unnecessary dimensions
    return cost


def mean_squared_error_prime(AL, Y):
    m = Y.shape[1]
    cost_derivative = -2 / m * (Y - AL)
    return cost_derivative


def cross_entropy_loss(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y*np.log(AL)) / m
    cost = np.squeeze(cost)
    return cost


def cross_entropy_loss_prime(AL, Y):
    m = Y.shape[1]
    cost_derivative = -Y/(AL * m)
    return cost_derivative


#
# ACTIVATION FUNCTIONS
#


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(Z, 0)
    return A


def sigmoid_prime(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


def relu_prime(Z):
    Z = np.where(Z > 0, 1, 0)
    return Z


def softmax(A):
    e_A = np.exp(A)
    return e_A / np.sum(e_A, axis=0, keepdims=False)


def softmax_prime(A):
    s = A.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


#
# UTILITY
#


def random_mini_batches(X, Y, batch_size):
    m = X.shape[1]
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    # Partition into mini-batches
    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        X_batch = X_shuffled[:, k * batch_size: (k + 1) * batch_size]
        Y_batch = Y_shuffled[:, k * batch_size: (k + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))

    # Handle last batch (if m % batch_size != 0)
    if m % batch_size != 0:
        X_batch = X_shuffled[:, num_complete_batches * batch_size:]
        Y_batch = Y_shuffled[:, num_complete_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches


def L_layer_model(X, Y, layers_dims, activations, learning_function="gradient_descent", loss_function="mean_squared_error", learning_rate=0.00075, use_softmax=True, num_iterations=1000, param_scale=0.01, batch_size=32, momentum_factor=0.2, print_cost=False):
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep(layers_dims, param_scale)
    if momentum_factor > 0:
        velocities = initialize_velocity(parameters)
    m = X.shape[1]  # Total training examples

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        mini_batches = random_mini_batches(X, Y, batch_size)
        cost_total = 0

        for X_batch, Y_batch in mini_batches:
            AL, caches = L_model_forward(X_batch, parameters, activations)
            AL = softmax(AL) if use_softmax else AL

            cost_total += compute_cost(AL, Y_batch, loss_function) * X_batch.shape[1]

            dAL = AL - Y_batch if use_softmax and loss_function == "cross_entropy_loss" else compute_cost_derivative(AL, Y_batch, loss_function)
            grads = L_model_backward(dAL, caches, activations)

            if momentum_factor > 0:
                parameters, velocities = update_parameters_velocity(parameters, grads, velocities, learning_rate, momentum_factor)
            else:
                parameters = update_parameters(parameters, grads, learning_rate)

        cost_avg = cost_total / m
        costs.append(cost_avg)

        # Print the cost every 100 iterations and for the last iteration
        if print_cost and (i % 10 == 0 or i == num_iterations - 1):
            print(f"Cost after iteration {i}: {cost_avg}")
    return parameters, costs


def predict_and_evaluate(X, parameters, activations, Y):
    AL, _ = L_model_forward(X, parameters, activations)

    predictions = np.argmax(AL, axis=0)

    true_labels = np.argmax(Y, axis=0)

    correct_guesses = np.sum(predictions == true_labels)
    total_samples = Y.shape[1]
    false_guesses = total_samples - correct_guesses
    accuracy = correct_guesses / total_samples

    return predictions, accuracy, correct_guesses, false_guesses