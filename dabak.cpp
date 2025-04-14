// Includes
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip> // Required for std::setprecision

// For CUDA (GPU) support -  Define this if you have CUDA enabled
// #define USE_CUDA 1

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// Define constants
const int MNIST_IMAGE_SIZE = 28 * 28; // 784
const int NUM_CLASSES = 10; // 0-9 digits
const int BATCH_SIZE = 128;   // Adjust as needed.  Larger batches can often improve GPU utilization.
const double LEARNING_RATE = 0.001; //  Adjust as needed
const double MOMENTUM = 0.9;       //  Momentum parameter
const int EPOCHS = 10;          // Number of training passes
const int NUM_THREADS = 8; // Number of threads for parallel processing (CPU)

// Function Declarations
std::vector<std::vector<double>> loadMNISTImages(const std::string& filename);
std::vector<unsigned char> loadMNISTLabels(const std::string& filename);
void preprocessData(std::vector<std::vector<double>>& images);
std::vector<std::vector<double>> createOneHotVectors(const std::vector<unsigned char>& labels);
void splitData(const std::vector<std::vector<double>>& images, const std::vector<std::vector<double>>& labels,
               std::vector<std::vector<double>>& trainImages, std::vector<std::vector<double>>& trainLabels,
               std::vector<std::vector<double>>& testImages, std::vector<std::vector<double>>& testLabels);

double calculateLoss(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);
double calculateAccuracy(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

// Helper function to print the shape of a matrix
void printShape(const std::vector<std::vector<double>>& matrix, const std::string& name) {
    std::cout << name << " shape: (" << matrix.size() << ", " << (matrix.empty() ? 0 : matrix[0].size()) << ")\n";
}

// Base Layer Class
class Layer {
public:
    virtual std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) = 0;
    virtual std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) = 0;
    virtual void updateWeights(double learningRate) = 0;
    virtual ~Layer() {} // Virtual destructor for proper inheritance cleanup

    // Helper functions for matrix operations
    static std::vector<std::vector<double>> matrixMultiply(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);
    static std::vector<std::vector<double>> matrixAdd(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);
    static std::vector<std::vector<double>> matrixSubtract(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);
    static std::vector<std::vector<double>> matrixMultiplyScalar(const std::vector<std::vector<double>>& matrix, double scalar);
    static std::vector<std::vector<double>> matrixTranspose(const std::vector<std::vector<double>>& matrix);
    static std::vector<std::vector<double>> applyFunction(const std::vector<std::vector<double>>& matrix, double (*func)(double));
    static std::vector<std::vector<double>> hadamardProduct(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);
    static double sum(const std::vector<std::vector<double>>& matrix);
    static std::vector<double> sumRows(const std::vector<std::vector<double>>& matrix);
    static std::vector<double> argmaxRows(const std::vector<std::vector<double>>& matrix);
};

// Implementation of matrix operations (static member functions)
std::vector<std::vector<double>> Layer::matrixMultiply(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    size_t m = a.size();     // Number of rows in a
    size_t n = a[0].size();  // Number of columns in a / Number of rows in b
    size_t p = b[0].size();  // Number of columns in b
    if (n != b.size()) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication.");
    }

    std::vector<std::vector<double>> result(m, std::vector<double>(p, 0.0)); // Initialize result matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < n; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<double>> Layer::matrixAdd(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Matrix dimensions do not match for addition.");
    }
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> Layer::matrixSubtract(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Matrix dimensions do not match for subtraction.");
    }
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> Layer::matrixMultiplyScalar(const std::vector<std::vector<double>>& matrix, double scalar) {
    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
    return result;
}

std::vector<std::vector<double>> Layer::matrixTranspose(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {};
    size_t m = matrix.size();
    size_t n = matrix[0].size();
    std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> Layer::applyFunction(const std::vector<std::vector<double>>& matrix, double (*func)(double)) {
    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i][j] = func(matrix[i][j]);
        }
    }
    return result;
}

std::vector<std::vector<double>> Layer::hadamardProduct(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Matrix dimensions do not match for Hadamard product.");
    }
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

double Layer::sum(const std::vector<std::vector<double>>& matrix) {
    double total = 0.0;
    for (const auto& row : matrix) {
        for (double val : row) {
            total += val;
        }
    }
    return total;
}

std::vector<double> Layer::sumRows(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {};
    size_t numRows = matrix.size();
    size_t numCols = matrix[0].size();
    std::vector<double> rowSums(numRows, 0.0);
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            rowSums[i] += matrix[i][j];
        }
    }
    return rowSums;
}
std::vector<double> Layer::argmaxRows(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {};
    size_t numRows = matrix.size();
    size_t numCols = matrix[0].size();
    std::vector<double> argmaxValues(numRows, 0.0); // Use double to be consistent
    for (size_t i = 0; i < numRows; ++i) {
        int maxIndex = 0;
        double maxValue = matrix[i][0];
        for (size_t j = 1; j < numCols; ++j) {
            if (matrix[i][j] > maxValue) {
                maxValue = matrix[i][j];
                maxIndex = j;
            }
        }
        argmaxValues[i] = static_cast<double>(maxIndex); // Store the index as a double
    }
    return argmaxValues;
}

// Fully Connected Layer
class FullyConnectedLayer : public Layer {
public:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> weightGradients;
    std::vector<std::vector<double>> biasGradients;
    std::vector<std::vector<double>> prevWeightGradients; // For momentum
    std::vector<std::vector<double>> prevBiasGradients;   // For momentum

    FullyConnectedLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize),
        weights(inputSize, std::vector<double>(outputSize)), biases(1, std::vector<double>(outputSize, 0.0)),
        weightGradients(inputSize, std::vector<double>(outputSize, 0.0)),
        biasGradients(1, std::vector<double>(outputSize, 0.0)),
        prevWeightGradients(inputSize, std::vector<double>(outputSize, 0.0)), // Initialize for momentum
        prevBiasGradients(1, std::vector<double>(outputSize, 0.0))           // Initialize for momentum
    {
        // Initialize weights with small random values using the He initialization method
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, std::sqrt(2.0 / inputSize)); // He initialization
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                weights[i][j] = dist(gen);
            }
        }
    }

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) override {
        input_ = input; // Store input for backward pass
        return matrixAdd(matrixMultiply(input, weights), biases);
    }

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) override {
        // Compute gradients
        weightGradients = matrixMultiply(matrixTranspose(input_), gradOutput);
        biasGradients = gradOutput; // Sum along the batch dimension.  gradOutput is (batch_size, outputSize)
        std::vector<std::vector<double>> inputGradient = matrixMultiply(gradOutput, matrixTranspose(weights));
        return inputGradient;
    }

    void updateWeights(double learningRate) override {
        // Update weights and biases using momentum
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                double weightUpdate = learningRate * weightGradients[i][j] + MOMENTUM * prevWeightGradients[i][j];
                weights[i][j] -= weightUpdate;
                prevWeightGradients[i][j] = weightUpdate; // Store for next iteration
            }
        }
        for (int j = 0; j < outputSize; ++j) {
            double biasUpdate = learningRate * biasGradients[0][j] + MOMENTUM * prevBiasGradients[0][j];
            biases[0][j] -= biasUpdate;
            prevBiasGradients[0][j] = biasUpdate;
        }
    }

private:
    std::vector<std::vector<double>> input_; // Store input for backward pass
};

// ReLU Activation Layer
class ReLULayer : public Layer {
public:
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) override {
        input_ = input; // Store input for backward pass
        return applyFunction(input, [](double x) { return x > 0 ? x : 0; });
    }

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) override {
        std::vector<std::vector<double>> inputGradient(input_.size(), std::vector<double>(input_[0].size(), 0.0));
        for (size_t i = 0; i < input_.size(); ++i) {
            for (size_t j = 0; j < input_[0].size(); ++j) {
                inputGradient[i][j] = gradOutput[i][j] * (input_[i][j] > 0 ? 1 : 0);
            }
        }
        return inputGradient;
    }

    void updateWeights(double learningRate) override {
        // ReLU has no weights to update
    }

private:
    std::vector<std::vector<double>> input_;
};

// Softmax Layer
class SoftmaxLayer : public Layer {
public:
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) override {
        // Numerical stability trick: subtract the maximum value
        std::vector<std::vector<double>> shiftedInput = input;
        for (auto& row : shiftedInput) {
            double maxVal = *std::max_element(row.begin(), row.end());
            for (double& val : row) {
                val -= maxVal;
            }
        }

        std::vector<std::vector<double>> expInput = applyFunction(shiftedInput, exp);
        output_ = expInput; // Store for backward pass
        std::vector<std::vector<double>> result(expInput.size(), std::vector<double>(expInput[0].size(), 0.0));
        for (size_t i = 0; i < expInput.size(); ++i) {
            double rowSum = 0.0;
            for (double val : expInput[i]) {
                rowSum += val;
            }
            for (size_t j = 0; j < expInput[0].size(); ++j) {
                result[i][j] = expInput[i][j] / rowSum;
            }
        }
        return result;
    }

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) override {
      // This is a combined Softmax and Cross-Entropy loss backward pass for efficiency
      //  If you were using a separate CrossEntropyLoss layer, this would be different.
      return gradOutput;
    }

    void updateWeights(double learningRate) override {
        // Softmax has no weights to update
    }

private:
    std::vector<std::vector<double>> output_; // Store output for backward pass
};

// Model Class
class Model {
public:
    std::vector<Layer*> layers;

    Model() {}

    void addLayer(Layer* layer) {
        layers.push_back(layer);
    }

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) {
        std::vector<std::vector<double>> output = input;
        for (Layer* layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) {
        std::vector<std::vector<double>> gradient = gradOutput;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradient = (*it)->backward(gradient);
        }
        return gradient;
    }

    void updateWeights(double learningRate) {
        for (Layer* layer : layers) {
            layer->updateWeights(learningRate);
        }
    }

     double calculateLoss(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) {
        double loss = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[0].size(); ++j) {
                // Cross-entropy loss:  Note:  This assumes the targets are one-hot encoded.
                loss -= targets[i][j] * log(predictions[i][j] + 1e-10); // Add a small epsilon to prevent log(0)
            }
        }
        return loss / predictions.size(); // Average loss over the batch
    }

    double calculateAccuracy(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) {
        std::vector<double> predictedLabels = Layer::argmaxRows(predictions);
        std::vector<double> targetLabels = Layer::argmaxRows(targets); // targets are one-hot
        int correct = 0;
        for (size_t i = 0; i < predictedLabels.size(); ++i) {
            if (predictedLabels[i] == targetLabels[i]) {
                correct++;
            }
        }
        return static_cast<double>(correct) / predictedLabels.size();
    }


    void train(const std::vector<std::vector<double>>& trainImages, const std::vector<std::vector<double>>& trainLabels,
               const std::vector<std::vector<double>>& testImages, const std::vector<std::vector<double>>& testLabels) {
        std::cout << "Training...\n";
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double epochLoss = 0.0;
            double epochAccuracy = 0.0;
            for (size_t i = 0; i < trainImages.size(); i += BATCH_SIZE) {
                // Get batch
                size_t end_index = std::min(i + BATCH_SIZE, trainImages.size());
                std::vector<std::vector<double>> batchImages(trainImages.begin() + i, trainImages.begin() + end_index);
                std::vector<std::vector<double>> batchLabels(trainLabels.begin() + i, trainLabels.begin() + end_index);

                // Forward pass
                std::vector<std::vector<double>> predictions = forward(batchImages);

                // Calculate loss and accuracy for this batch
                double batchLoss = calculateLoss(predictions, batchLabels);
                double batchAccuracy = calculateAccuracy(predictions, batchLabels);
                epochLoss += batchLoss * batchImages.size(); // Weighted sum for correct averaging
                epochAccuracy += batchAccuracy * batchImages.size();

                // Backward pass
                backward(matrixSubtract(predictions, batchLabels)); // Simplified for softmax/cross-entropy combination

                // Update weights
                updateWeights(LEARNING_RATE / batchImages.size()); // Divide by batch size

                if ((i / BATCH_SIZE) % 100 == 0) {
                    std::cout << "  Epoch " << epoch + 1 << ", Batch " << i / BATCH_SIZE
                              << ", Loss: " << std::fixed << std::setprecision(4) << batchLoss
                              << ", Accuracy: " << std::fixed << std::setprecision(4) << batchAccuracy << "\n";
                }
            }
            // Calculate average loss and accuracy for the epoch
            epochLoss /= trainImages.size();
            epochAccuracy /= trainImages.size();
            std::cout << "Epoch " << epoch + 1 << " completed, Loss: " << std::fixed << std::setprecision(4) << epochLoss
                      << ", Accuracy: " << std::fixed << std::setprecision(4) << epochAccuracy << "\n";

            // Evaluate on test set at the end of each epoch
            evaluate(testImages, testLabels);
        }
    }

    void evaluate(const std::vector<std::vector<double>>& testImages, const std::vector<std::vector<double>>& testLabels) {
        std::cout << "Evaluating on test set...\n";
        double testLoss = 0.0;
        double testAccuracy = 0.0;
        for (size_t i = 0; i < testImages.size(); i += BATCH_SIZE) {
             size_t end_index = std::min(i + BATCH_SIZE, testImages.size());
            std::vector<std::vector<double>> batchImages(testImages.begin() + i, testImages.begin() + end_index);
            std::vector<std::vector<double>> batchLabels(testLabels.begin() + i, testLabels.begin() + end_index);
            std::vector<std::vector<double>> predictions = forward(batchImages);
            testLoss += calculateLoss(predictions, batchLabels) * batchImages.size();
            testAccuracy += calculateAccuracy(predictions, batchLabels) * batchImages.size();
        }
        testLoss /= testImages.size();
        testAccuracy /= testImages.size();
        std::cout << "Test Loss: " << std::fixed << std::setprecision(4) << testLoss
                  << ", Test Accuracy: " << std::fixed << std::setprecision(4) << testAccuracy << "\n";
    }

};

// Function to load MNIST images from the specified file.
std::vector<std::vector<double>> loadMNISTImages(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open image file: " + filename);
    }

    int magicNumber = 0;
    int numImages = 0;
    int numRows = 0;
    int numCols = 0;

    // Read theheader information
    file.read(reinterpret_cast<char*>(&magicNumber), 4); // Magic number
    file.read(reinterpret_cast<char*>(&numImages), 4);     // Number of images
    file.read(reinterpret_cast<char*>(&numRows), 4);       // Number of rows
    file.read(reinterpret_cast<char*>(&numCols), 4);       // Number of columns

    // Convert from big-endian to little-endian
    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);
    numRows = __builtin_bswap32(numRows);
    numCols = __builtin_bswap32(numCols);

    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid magic number in image file.");
    }
    if (numRows != 28 || numCols != 28) {
        throw std::runtime_error("Invalid image dimensions. Expected 28x28.");
    }

    // Read the image data
    std::vector<std::vector<double>> images(numImages, std::vector<double>(numRows * numCols));
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < numRows * numCols; ++j) {
            unsigned char pixelValue;
            file.read(reinterpret_cast<char*>(&pixelValue), 1);
            images[i][j] = static_cast<double>(pixelValue); // Convert to double
        }
    }
    file.close();
    return images;
}

// Function to load MNIST labels from the specified file.
std::vector<unsigned char> loadMNISTLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open label file: " + filename);
    }

    int magicNumber = 0;
    int numItems = 0;

    // Read the header information
    file.read(reinterpret_cast<char*>(&magicNumber), 4); // Magic number
    file.read(reinterpret_cast<char*>(&numItems), 4);    // Number of items

    // Convert from big-endian to little-endian
    magicNumber = __builtin_bswap32(magicNumber);
    numItems = __builtin_bswap32(numItems);

    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid magic number in label file.");
    }

    // Read the label data
    std::vector<unsigned char> labels(numItems);
    file.read(reinterpret_cast<char*>(labels.data()), numItems);
    file.close();
    return labels;
}

// Function to preprocess the image data.
void preprocessData(std::vector<std::vector<double>>& images) {
    // Normalize pixel values to the range [0, 1] and subtract the mean.
    double mean = 0.0;
    for (const auto& image : images) {
        for (double pixelValue : image) {
            mean += pixelValue;
        }
    }
    mean /= (images.size() * images[0].size()); // Calculate the mean pixel value

    double stdDev = 0.0;
     for (const auto& image : images) {
        for (double pixelValue : image) {
            stdDev += std::pow(pixelValue - mean, 2);
        }
    }
    stdDev = std::sqrt(stdDev / (images.size() * images[0].size()));

    const double epsilon = 1e-7; // small constant to prevent division by zero.

    for (auto& image : images) {
        for (double& pixelValue : image) {
            pixelValue = (pixelValue - mean) / (stdDev + epsilon);
            pixelValue = (pixelValue / 255.0); // Normalize to [0,1]
        }
    }
}

// Function to create one-hot encoded vectors from the labels.
std::vector<std::vector<double>> createOneHotVectors(const std::vector<unsigned char>& labels) {
    std::vector<std::vector<double>> oneHotVectors(labels.size(), std::vector<double>(NUM_CLASSES, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        oneHotVectors[i][labels[i]] = 1.0;
    }
    return oneHotVectors;
}

// Function to split the data into training and testing sets.
void splitData(const std::vector<std::vector<double>>& images, const std::vector<std::vector<double>>& labels,
               std::vector<std::vector<double>>& trainImages, std::vector<std::vector<double>>& trainLabels,
               std::vector<std::vector<double>>& testImages, std::vector<std::vector<double>>& testLabels) {
    // Simple 80/20 split
    size_t splitIndex = static_cast<size_t>(images.size() * 0.8);
    trainImages = std::vector<std::vector<double>>(images.begin(), images.begin() + splitIndex);
    trainLabels = std::vector<std::vector<double>>(labels.begin(), labels.begin() + splitIndex);
    testImages = std::vector<std::vector<double>>(images.begin() + splitIndex, images.end());
    testLabels = std::vector<std::vector<double>>(labels.begin() + splitIndex, labels.end());
}

int main() {
    try {
        // Load the MNIST dataset
        std::vector<std::vector<double>> images = loadMNISTImages("train-images.idx3-ubyte"); // Path to training images
        std::vector<unsigned char> labels = loadMNISTLabels("train-labels.idx1-ubyte");   // Path to training labels

        // Preprocess the data
        preprocessData(images);

        // Create one-hot encoded vectors for the labels
        std::vector<std::vector<double>> oneHotLabels = createOneHotVectors(labels);

        // Split the data into training and testing sets
        std::vector<std::vector<double>> trainImages, trainLabels, testImages, testLabels;
        splitData(images, oneHotLabels, trainImages, trainLabels, testImages, testLabels);

        // Create the model
        Model model;
        model.addLayer(new FullyConnectedLayer(MNIST_IMAGE_SIZE, 256));
        model.addLayer(new ReLULayer());
        model.addLayer(new FullyConnectedLayer(256, 128));
        model.addLayer(new ReLULayer());
        model.addLayer(new FullyConnectedLayer(128, NUM_CLASSES));
        model.addLayer(new SoftmaxLayer());

        // Print the shapes of the training and testing data.
        printShape(trainImages, "Train Images");
        printShape(trainLabels, "Train Labels");
        printShape(testImages, "Test Images");
        printShape(testLabels, "Test Labels");

        // Train the model
        model.train(trainImages, trainLabels, testImages, testLabels);

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
