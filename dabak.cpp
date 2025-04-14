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
#include <stdexcept> // Required for std::runtime_error

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
const int NUM_THREADS = 8; // Number of threads for parallel processing (CPU) - Adjust as needed

// Function Declarations
std::vector<std::vector<double>> loadMNISTImages(const std::string& filename);
std::vector<unsigned char> loadMNISTLabels(const std::string& filename);
void preprocessData(std::vector<std::vector<double>>& images);
std::vector<std::vector<double>> createOneHotVectors(const std::vector<unsigned char>& labels);
void splitData(const std::vector<std::vector<double>>& images, const std::vector<std::vector<double>>& labels,
               std::vector<std::vector<double>>& trainImages, std::vector<std::vector<double>>& trainLabels,
               std::vector<std::vector<double>>& testImages, std::vector<std::vector<double>>& testLabels);

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

    // Helper functions for matrix operations (static)
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
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Matrix multiplication with empty matrix.");
    }
    size_t m = a.size();     // Number of rows in a
    size_t n = a[0].size();  // Number of columns in a / Number of rows in b
    size_t p = b[0].size();  // Number of columns in b
    if (n != b.size()) {
         throw std::runtime_error("Matrix dimensions do not match for multiplication. A(" + std::to_string(m) + "," + std::to_string(n) + ") B(" + std::to_string(b.size()) + "," + std::to_string(p) + ")");
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
     if (a.empty() || b.empty()) {
        throw std::runtime_error("Matrix addition with empty matrix.");
    }
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Matrix dimensions do not match for addition. A(" + std::to_string(a.size()) + "," + std::to_string(a[0].size()) + ") B(" + std::to_string(b.size()) + "," + std::to_string(b[0].size()) + ")");
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
     if (a.empty() || b.empty()) {
        throw std::runtime_error("Matrix subtraction with empty matrix.");
    }
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Matrix dimensions do not match for subtraction. A(" + std::to_string(a.size()) + "," + std::to_string(a[0].size()) + ") B(" + std::to_string(b.size()) + "," + std::to_string(b[0].size()) + ")");
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
    if (matrix.empty()) return {};
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
    if (matrix.empty()) return {};
    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i][j] = func(matrix[i][j]);
        }
    }
    return result;
}

std::vector<std::vector<double>> Layer::hadamardProduct(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    if (a.empty() || b.empty()) {
        throw std::runtime_error("Hadamard product with empty matrix.");
    }
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::runtime_error("Matrix dimensions do not match for Hadamard product. A(" + std::to_string(a.size()) + "," + std::to_string(a[0].size()) + ") B(" + std::to_string(b.size()) + "," + std::to_string(b[0].size()) + ")");
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
    std::vector<std::vector<double>> biasGradients; // Should be (1, outputSize)
    std::vector<std::vector<double>> prevWeightGradients; // For momentum
    std::vector<std::vector<double>> prevBiasGradients;   // For momentum

    FullyConnectedLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize),
        weights(inputSize, std::vector<double>(outputSize)), biases(1, std::vector<double>(outputSize, 0.0)),
        weightGradients(inputSize, std::vector<double>(outputSize, 0.0)),
        biasGradients(1, std::vector<double>(outputSize, 0.0)), // Correct size (1, outputSize)
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
        input_ = input; // Store input for backward pass (batch_size, inputSize)
        // input (batch_size, inputSize) * weights (inputSize, outputSize) -> (batch_size, outputSize)
        std::vector<std::vector<double>> output = matrixMultiply(input, weights);
        // Add biases (broadcasted)
        for(size_t i = 0; i < output.size(); ++i) {
            for(size_t j = 0; j < output[0].size(); ++j) {
                output[i][j] += biases[0][j];
            }
        }
        return output;
    }

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) override {
        // gradOutput shape: (batch_size, outputSize)
        // input_ shape: (batch_size, inputSize)
        // weights shape: (inputSize, outputSize)

        // Compute weight gradients: input_T * gradOutput
        // (inputSize, batch_size) * (batch_size, outputSize) -> (inputSize, outputSize)
        weightGradients = matrixMultiply(matrixTranspose(input_), gradOutput);

        // Compute bias gradients: sum gradOutput along the batch dimension (axis 0)
        // Result should be (1, outputSize)
        biasGradients = std::vector<std::vector<double>>(1, std::vector<double>(outputSize, 0.0));
        for(size_t j = 0; j < outputSize; ++j) {
            for(size_t i = 0; i < gradOutput.size(); ++i) {
                biasGradients[0][j] += gradOutput[i][j];
            }
        }

        // Compute input gradient: gradOutput * weights_T
        // (batch_size, outputSize) * (outputSize, inputSize) -> (batch_size, inputSize)
        std::vector<std::vector<double>> inputGradient = matrixMultiply(gradOutput, matrixTranspose(weights));
        return inputGradient;
    }

    void updateWeights(double learningRate) override {
        // Update weights and biases using momentum
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                // Note: Gradients are summed over the batch, so learning rate adjustment might be needed
                // depending on whether you average the loss or sum it. Here we assume average loss.
                double weightUpdate = learningRate * weightGradients[i][j] + MOMENTUM * prevWeightGradients[i][j];
                weights[i][j] -= weightUpdate;
                prevWeightGradients[i][j] = weightUpdate; // Store for next iteration
            }
        }
        for (int j = 0; j < outputSize; ++j) {
             // Note: Bias gradients are also summed over the batch.
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
        if (input_.empty()) throw std::runtime_error("ReLU backward called with empty input cache.");
        std::vector<std::vector<double>> inputGradient(input_.size(), std::vector<double>(input_[0].size(), 0.0));
        for (size_t i = 0; i < input_.size(); ++i) {
            for (size_t j = 0; j < input_[0].size(); ++j) {
                inputGradient[i][j] = gradOutput[i][j] * (input_[i][j] > 0 ? 1.0 : 0.0); // Use 1.0 for clarity
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
        if (input.empty()) return {};
        // Numerical stability trick: subtract the maximum value per row
        std::vector<std::vector<double>> shiftedInput = input;
        for (auto& row : shiftedInput) {
            if (row.empty()) continue; // Skip empty rows
            double maxVal = *std::max_element(row.begin(), row.end());
            for (double& val : row) {
                val -= maxVal;
            }
        }

        std::vector<std::vector<double>> expInput = applyFunction(shiftedInput, exp);
        output_ = std::vector<std::vector<double>>(expInput.size(), std::vector<double>(expInput[0].size(), 0.0)); // Initialize output_
        for (size_t i = 0; i < expInput.size(); ++i) {
            double rowSum = 0.0;
            for (double val : expInput[i]) {
                rowSum += val;
            }
            // Handle potential division by zero if rowSum is very small or zero
            if (rowSum < 1e-10) rowSum = 1e-10;
            for (size_t j = 0; j < expInput[0].size(); ++j) {
                output_[i][j] = expInput[i][j] / rowSum;
            }
        }
        return output_; // Return the calculated softmax probabilities
    }

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& gradOutput) override {
      // This backward pass assumes it's combined with Cross-Entropy Loss.
      // The gradient passed *in* (gradOutput) should be (predictions - targets).
      // If Softmax were standalone, its Jacobian calculation would be more complex.
      // Here, we just pass the gradient through, as the combined derivative simplifies.
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

    ~Model() { // Destructor to clean up dynamically allocated layers
        for (Layer* layer : layers) {
            delete layer;
        }
        layers.clear();
    }

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
        return gradient; // Return the gradient w.r.t the model's input (usually not needed)
    }

    void updateWeights(double learningRate) {
        for (Layer* layer : layers) {
            layer->updateWeights(learningRate);
        }
    }

     // Cross-Entropy Loss Calculation
     double calculateLoss(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) {
        if (predictions.empty() || targets.empty() || predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) {
             throw std::runtime_error("Invalid dimensions for loss calculation.");
        }
        double loss = 0.0;
        const double epsilon = 1e-10; // Small value to prevent log(0)
        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[0].size(); ++j) {
                // Cross-entropy loss: - sum(target * log(prediction))
                loss -= targets[i][j] * log(predictions[i][j] + epsilon);
            }
        }
        return loss / predictions.size(); // Average loss over the batch
    }

    // Accuracy Calculation
    double calculateAccuracy(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) {
         if (predictions.empty() || targets.empty() || predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) {
             throw std::runtime_error("Invalid dimensions for accuracy calculation.");
        }
        std::vector<double> predictedLabels = Layer::argmaxRows(predictions);
        std::vector<double> targetLabels = Layer::argmaxRows(targets); // targets are one-hot
        int correct = 0;
        for (size_t i = 0; i < predictedLabels.size(); ++i) {
            // Compare the predicted index with the target index
            if (static_cast<int>(predictedLabels[i]) == static_cast<int>(targetLabels[i])) {
                correct++;
            }
        }
        return static_cast<double>(correct) / predictedLabels.size();
    }


    void train(const std::vector<std::vector<double>>& trainImages, const std::vector<std::vector<double>>& trainLabels,
               const std::vector<std::vector<double>>& testImages, const std::vector<std::vector<double>>& testLabels) {
        std::cout << "Training...\n";
        size_t numTrainSamples = trainImages.size();
        if (numTrainSamples == 0) {
            std::cerr << "Warning: Training dataset is empty." << std::endl;
            return;
        }

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double epochLoss = 0.0;
            double epochAccuracy = 0.0;
            int batchesProcessed = 0;

            // Shuffle training data (optional but recommended)
            // std::vector<size_t> indices(numTrainSamples);
            // std::iota(indices.begin(), indices.end(), 0);
            // std::random_shuffle(indices.begin(), indices.end()); // Requires <algorithm> and <numeric>

            for (size_t i = 0; i < numTrainSamples; i += BATCH_SIZE) {
                // Get batch
                size_t end_index = std::min(i + BATCH_SIZE, numTrainSamples);
                size_t currentBatchSize = end_index - i;
                if (currentBatchSize == 0) continue;

                std::vector<std::vector<double>> batchImages;
                std::vector<std::vector<double>> batchLabels;
                batchImages.reserve(currentBatchSize);
                batchLabels.reserve(currentBatchSize);

                for(size_t k = i; k < end_index; ++k) {
                    // Use shuffled indices if implemented: batchImages.push_back(trainImages[indices[k]]);
                    batchImages.push_back(trainImages[k]);
                    batchLabels.push_back(trainLabels[k]);
                }


                // Forward pass
                std::vector<std::vector<double>> predictions = forward(batchImages);

                // Calculate loss and accuracy for this batch
                double batchLoss = calculateLoss(predictions, batchLabels);
                double batchAccuracy = calculateAccuracy(predictions, batchLabels);
                epochLoss += batchLoss * currentBatchSize; // Accumulate total loss
                epochAccuracy += batchAccuracy * currentBatchSize; // Accumulate total correct predictions scaled by batch size
                batchesProcessed++;

                // Backward pass: Gradient of CrossEntropy+Softmax is (predictions - targets)
                // Ensure dimensions match before subtraction
                if (predictions.size() != batchLabels.size() || (!predictions.empty() && predictions[0].size() != batchLabels[0].size())) {
                     throw std::runtime_error("Dimension mismatch between predictions and labels before backward pass.");
                }
                std::vector<std::vector<double>> lossGradient = Layer::matrixSubtract(predictions, batchLabels); // *** FIX HERE ***

                // Propagate gradient backwards through the network
                backward(lossGradient);

                // Update weights (apply gradients)
                // Learning rate is often scaled by 1/batch_size when loss is averaged per batch
                updateWeights(LEARNING_RATE); // Pass the base learning rate

                // Print progress periodically
                if (batchesProcessed % 100 == 0) {
                    std::cout << "  Epoch " << epoch + 1 << ", Batch " << batchesProcessed
                              << ", Loss: " << std::fixed << std::setprecision(4) << batchLoss
                              << ", Accuracy: " << std::fixed << std::setprecision(4) << batchAccuracy << "\n";
                }
            }
            // Calculate average loss and accuracy for the epoch
            epochLoss /= numTrainSamples;
            epochAccuracy /= numTrainSamples;
            std::cout << "Epoch " << epoch + 1 << " completed, Avg Loss: " << std::fixed << std::setprecision(4) << epochLoss
                      << ", Avg Accuracy: " << std::fixed << std::setprecision(4) << epochAccuracy << "\n";

            // Evaluate on test set at the end of each epoch
            evaluate(testImages, testLabels);
            std::cout << "----------------------------------------\n"; // Separator
        }
    }

    void evaluate(const std::vector<std::vector<double>>& testImages, const std::vector<std::vector<double>>& testLabels) {
        std::cout << "Evaluating on test set...\n";
        size_t numTestSamples = testImages.size();
         if (numTestSamples == 0) {
            std::cerr << "Warning: Test dataset is empty." << std::endl;
            return;
        }
        double testLoss = 0.0;
        double testAccuracy = 0.0;
        for (size_t i = 0; i < numTestSamples; i += BATCH_SIZE) {
             size_t end_index = std::min(i + BATCH_SIZE, numTestSamples);
             size_t currentBatchSize = end_index - i;
             if (currentBatchSize == 0) continue;

            std::vector<std::vector<double>> batchImages;
            std::vector<std::vector<double>> batchLabels;
            batchImages.reserve(currentBatchSize);
            batchLabels.reserve(currentBatchSize);

            for(size_t k = i; k < end_index; ++k) {
                 batchImages.push_back(testImages[k]);
                 batchLabels.push_back(testLabels[k]);
            }

            std::vector<std::vector<double>> predictions = forward(batchImages);
            testLoss += calculateLoss(predictions, batchLabels) * currentBatchSize;
            testAccuracy += calculateAccuracy(predictions, batchLabels) * currentBatchSize;
        }
        testLoss /= numTestSamples;
        testAccuracy /= numTestSamples;
        std::cout << "Test Loss: " << std::fixed << std::setprecision(4) << testLoss
                  << ", Test Accuracy: " << std::fixed << std::setprecision(4) << testAccuracy << "\n";
    }

};

// --- MNIST Loading and Preprocessing Functions ---

// Function to swap endianness (MNIST data is big-endian)
int swapEndian(int val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


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

    // Read the header information
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numImages), 4);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&numCols), 4);

    // Convert from big-endian to host byte order (usually little-endian)
    magicNumber = swapEndian(magicNumber);
    numImages = swapEndian(numImages);
    numRows = swapEndian(numRows);
    numCols = swapEndian(numCols);


    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid magic number in image file. Expected 2051, got " + std::to_string(magicNumber));
    }
    if (numRows != 28 || numCols != 28) {
         throw std::runtime_error("Invalid image dimensions. Expected 28x28, got " + std::to_string(numRows) + "x" + std::to_string(numCols));
    }
    if (numImages <= 0) {
         throw std::runtime_error("Invalid number of images read from file: " + std::to_string(numImages));
    }


    // Read the image data
    int imageSize = numRows * numCols;
    std::vector<std::vector<double>> images(numImages, std::vector<double>(imageSize));
    std::vector<unsigned char> buffer(imageSize); // Read row by row into buffer first

    for (int i = 0; i < numImages; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), imageSize);
        if (!file) {
             throw std::runtime_error("Error reading image data for image " + std::to_string(i) + " from file: " + filename);
        }
        for (int j = 0; j < imageSize; ++j) {
            images[i][j] = static_cast<double>(buffer[j]); // Convert to double
        }
    }
    file.close();
    std::cout << "Loaded " << numImages << " images (" << numRows << "x" << numCols << ") from " << filename << std::endl;
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
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numItems), 4);

    // Convert from big-endian to host byte order
    magicNumber = swapEndian(magicNumber);
    numItems = swapEndian(numItems);

    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid magic number in label file. Expected 2049, got " + std::to_string(magicNumber));
    }
     if (numItems <= 0) {
         throw std::runtime_error("Invalid number of labels read from file: " + std::to_string(numItems));
    }


    // Read the label data
    std::vector<unsigned char> labels(numItems);
    file.read(reinterpret_cast<char*>(labels.data()), numItems);
     if (!file) {
         throw std::runtime_error("Error reading label data from file: " + filename);
    }
    file.close();
    std::cout << "Loaded " << numItems << " labels from " << filename << std::endl;
    return labels;
}

// Function to preprocess the image data (Normalization).
void preprocessData(std::vector<std::vector<double>>& images) {
    if (images.empty()) return;
    // Normalize pixel values to the range [0, 1]
    for (auto& image : images) {
        for (double& pixelValue : image) {
            pixelValue /= 255.0;
        }
    }
     // Optional: Subtract mean and divide by std deviation (calculated per dataset)
    // double mean = 0.1307; // Pre-calculated mean for MNIST
    // double stdDev = 0.3081; // Pre-calculated std deviation for MNIST
    // for (auto& image : images) {
    //     for (double& pixelValue : image) {
    //         pixelValue = (pixelValue - mean) / stdDev;
    //     }
    // }
    std::cout << "Preprocessed " << images.size() << " images (Normalization)." << std::endl;
}

// Function to create one-hot encoded vectors from the labels.
std::vector<std::vector<double>> createOneHotVectors(const std::vector<unsigned char>& labels) {
    if (labels.empty()) return {};
    std::vector<std::vector<double>> oneHotVectors(labels.size(), std::vector<double>(NUM_CLASSES, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] >= NUM_CLASSES) {
             throw std::runtime_error("Label value " + std::to_string(labels[i]) + " is out of range [0, " + std::to_string(NUM_CLASSES - 1) + "]");
        }
        oneHotVectors[i][labels[i]] = 1.0;
    }
     std::cout << "Created " << oneHotVectors.size() << " one-hot label vectors." << std::endl;
    return oneHotVectors;
}

// Function to split the data into training and testing sets.
void splitData(const std::vector<std::vector<double>>& images, const std::vector<std::vector<double>>& labels,
               std::vector<std::vector<double>>& trainImages, std::vector<std::vector<double>>& trainLabels,
               std::vector<std::vector<double>>& testImages, std::vector<std::vector<double>>& testLabels,
               double trainSplitRatio = 0.8) { // Allow specifying split ratio
    if (images.size() != labels.size()) {
        throw std::runtime_error("Number of images and labels must match for splitting.");
    }
    if (trainSplitRatio <= 0.0 || trainSplitRatio >= 1.0) {
         throw std::runtime_error("Train split ratio must be between 0 and 1.");
    }

    size_t totalSize = images.size();
    size_t splitIndex = static_cast<size_t>(totalSize * trainSplitRatio);
    if (splitIndex == 0 || splitIndex == totalSize) {
        std::cerr << "Warning: Split resulted in an empty training or testing set. Adjust split ratio or dataset size." << std::endl;
    }


    // Create indices and shuffle them for a random split
    std::vector<size_t> indices(totalSize);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g); // Shuffle indices randomly

    // Reserve space for efficiency
    trainImages.reserve(splitIndex);
    trainLabels.reserve(splitIndex);
    testImages.reserve(totalSize - splitIndex);
    testLabels.reserve(totalSize - splitIndex);

    // Assign data based on shuffled indices
    for (size_t i = 0; i < splitIndex; ++i) {
        trainImages.push_back(images[indices[i]]);
        trainLabels.push_back(labels[indices[i]]);
    }
    for (size_t i = splitIndex; i < totalSize; ++i) {
        testImages.push_back(images[indices[i]]);
        testLabels.push_back(labels[indices[i]]);
    }
    std::cout << "Split data: " << trainImages.size() << " training samples, " << testImages.size() << " testing samples." << std::endl;
}

// --- Main Function ---
int main() {
    try {
        // Define file paths (adjust if necessary)
        const std::string trainImagesPath = "train-images.idx3-ubyte";
        const std::string trainLabelsPath = "train-labels.idx1-ubyte";
        // Optional: Load separate test set if available
        // const std::string testImagesPath = "t10k-images.idx3-ubyte";
        // const std::string testLabelsPath = "t10k-labels.idx1-ubyte";

        // Load the MNIST training dataset
        std::vector<std::vector<double>> images = loadMNISTImages(trainImagesPath);
        std::vector<unsigned char> labels = loadMNISTLabels(trainLabelsPath);

        // Preprocess the data
        preprocessData(images); // Normalize images

        // Create one-hot encoded vectors for the labels
        std::vector<std::vector<double>> oneHotLabels = createOneHotVectors(labels);

        // Split the loaded training data into training and validation sets
        std::vector<std::vector<double>> trainImages, trainLabels, valImages, valLabels;
        splitData(images, oneHotLabels, trainImages, trainLabels, valImages, valLabels, 0.85); // e.g., 85% train, 15% validation

        // --- Build the Model ---
        Model model;
        model.addLayer(new FullyConnectedLayer(MNIST_IMAGE_SIZE, 256)); // Input: 784 features
        model.addLayer(new ReLULayer());
        model.addLayer(new FullyConnectedLayer(256, 128));
        model.addLayer(new ReLULayer());
        model.addLayer(new FullyConnectedLayer(128, NUM_CLASSES));      // Output: 10 classes
        model.addLayer(new SoftmaxLayer()); // Softmax for probability distribution

        // Print the shapes of the training and validation data.
        printShape(trainImages, "Train Images");
        printShape(trainLabels, "Train Labels");
        printShape(valImages, "Validation Images");
        printShape(valLabels, "Validation Labels");

        // Train the model using the training set and validate on the validation set
        model.train(trainImages, trainLabels, valImages, valLabels);

        std::cout << "\nTraining finished." << std::endl;

        // Optional: If you have a separate test set (t10k files), load and evaluate it here
        /*
        try {
            std::cout << "\nLoading final test set..." << std::endl;
            std::vector<std::vector<double>> finalTestImages = loadMNISTImages(testImagesPath);
            std::vector<unsigned char> finalTestLabelsRaw = loadMNISTLabels(testLabelsPath);
            preprocessData(finalTestImages);
            std::vector<std::vector<double>> finalTestLabels = createOneHotVectors(finalTestLabelsRaw);
            printShape(finalTestImages, "Final Test Images");
            printShape(finalTestLabels, "Final Test Labels");
            std::cout << "\nEvaluating on final test set..." << std::endl;
            model.evaluate(finalTestImages, finalTestLabels);
        } catch (const std::runtime_error& e) {
            std::cerr << "Could not load or evaluate final test set: " << e.what() << std::endl;
            std::cerr << "Ensure '" << testImagesPath << "' and '" << testLabelsPath << "' are present." << std::endl;
        }
        */


    } catch (const std::runtime_error& e) {
        std::cerr << "\n--- An error occurred ---\n" << e.what() << std::endl;
        return 1; // Indicate failure
    } catch (...) { // Catch any other unexpected exceptions
        std::cerr << "\n--- An unexpected error occurred ---" << std::endl;
        return 1;
    }

    return 0; // Indicate success
}
