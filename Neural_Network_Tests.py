# Main Libraries
import matplotlib.pyplot as plt
import numpy as np
import Deep_Neural_Network as dnn
import Metrics_and_Visualizations as metrics

# Import Datasets and Dataset Loaders
#from testCases_v2 import *
#from public_tests import *
#from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset,load_extra_datasets
from sklearn.datasets import fetch_openml

from Metrics_and_Visualizations import plot_confusion_matrix, plot_training_history, show_error_samples, \
    plot_class_distribution

# Load MNIST (this may take a moment on first run)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Data Pre-processing and labels
X, y = mnist["data"], mnist["target"]
m = y.shape[0]
y_vector = np.zeros((10, m))
for i in range(m):
    y_vector[int(y[i]), i] = 1
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X_reshaped = X.T.reshape(784, -1)
X_reshaped = X_reshaped / 255.0 # Normalize the image values (very important)


X_train = X_reshaped[:, :60000]  # (784, 60000)
X_test = X_reshaped[:, 60000:]   # (784, 10000)

y_train = y_vector[:, :60000]  # (10, 60000)
y_test = y_vector[:, 60000:].astype(np.int64)  # (10, 10000)
y_test_labels = np.astype(y[60000:], np.int64)

# Set Model Hyperparameters
layer_dims = [784, 24, 12, 10]
activations = ["relu", "relu", "relu", "sigmoid"]
#param_scale=0.01, learning_rate=0.00075, batch_size=32, num_iterations=1000,

# Run the Model and Make Predictions
parameters, costs = dnn.L_layer_model(X_train, y_train, layer_dims, activations, param_scale=0.01, learning_rate=0.00081, batch_size=32, num_iterations=10, print_cost=True)
predictions, accuracy, correct_matches, false_matches = dnn.predict_and_evaluate(X_test, parameters, activations, y_test)

loss:dict[str, int] = dict(loss=costs, val_loss=0, accuracy=accuracy, val_accuracy=0)

# See Metrics
print(f"Accuracy: {100*accuracy}%")
print(f"Number of Correct Predictions: {correct_matches}")
print(f"Number of False Predictions: {false_matches}")
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Advice: use y_test_labels instead of y_test if you don't want to get a headache while debugging
plot_training_history(loss)
plot_confusion_matrix(y_test_labels, predictions, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
show_error_samples(X_test, y_test_labels, predictions, class_names, 5)
plot_class_distribution(y_test_labels, predictions, class_names)


