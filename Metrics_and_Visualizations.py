from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.show()


def show_error_samples(X, y_true, y_pred, class_names, num_samples=5):
    errors = np.where(y_true != y_pred)[0]
    sample_indices = np.random.choice(errors, num_samples)

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(sample_indices):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[:,idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}')
        plt.axis('off')
    plt.show()


def plot_feature_importance(weights, feature_names, top_n=20):
    importance = np.mean(np.abs(weights), axis=1)
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance (First Layer Weights)')
    plt.barh(range(top_n), importance[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Mean Absolute Weight')
    plt.show()


def plot_class_distribution(y_true, y_pred, classes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # True distribution
    sns.countplot(x=y_true, ax=ax1, order=classes)
    ax1.set_title('True Class Distribution')

    # Predicted distribution
    sns.countplot(x=y_pred, ax=ax2, order=classes)
    ax2.set_title('Predicted Class Distribution')

    plt.show()


def plot_weight_recognition(weights):
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(weights.shape):
        #plt.subplot(1, num_samples, i + 1)
        plt.imshow(weights[:, idx].reshape(28, 28), cmap='red')
        plt.title(f'Neuron {i}')
        plt.axis('off')
    plt.show()