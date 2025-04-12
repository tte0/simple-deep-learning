import Deep_Neural_Network as dnn
import Metrics_and_Visualizations as metrics
from sklearn.datasets import fetch_openml
import numpy as np
from Metrics_and_Visualizations import *

# Load Datasets
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X, y = fashion_mnist["data"], fashion_mnist["target"]

m = y.shape[0]
y_vector = np.zeros((10, m))
for i in range(m):
    y_vector[int(y[i]), i] = 1
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Fashion-MNIST labels

X_reshaped = X.T.reshape(784, -1) / 255.0  # Normalize

# Split train/test
X_train = X_reshaped[:, :60000]
X_test = X_reshaped[:, 60000:]
y_train = y_vector[:, :60000]
y_test = y_vector[:, 60000:].astype(np.int64)
y_test_labels = np.array(y[60000:], dtype=np.int64)

layer_dims = [784, 24, 12, 10]
activations = ["relu", "relu", "relu", "sigmoid"]

parameters, costs = dnn.L_layer_model(X_train, y_train, layer_dims, activations,
                                     param_scale=0.01, learning_rate=0.00081,
                                     batch_size=32, num_iterations=100, print_cost=True)
predictions, accuracy, correct_matches, false_matches = dnn.predict_and_evaluate(
    X_test, parameters, activations, y_test)

loss:dict[str, int] = dict(loss=costs, val_loss=0, accuracy=accuracy, val_accuracy=0)


# See results (FUN PART)
plot_training_history(loss)
plot_confusion_matrix(y_test_labels, predictions, classes=class_names)
show_error_samples(X_test, y_test_labels, predictions, class_names, 5)