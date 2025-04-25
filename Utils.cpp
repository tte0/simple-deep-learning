#include <vector>
#include <random>
#include "Utils.h"

std::random_device rd;
std::mt19937 rng(rd());

std::vector<std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>>>
random_mini_batches(std::vector<std::vector<double>> X, std::vector<std::vector<double>> Y, size_t batch_size){
    size_t n = X.size();
    size_t m = X[0].size();

    // Shuffle (X, Y)
    for(size_t i = n; i > 0; i--){
        size_t j = rng() % i;
        swap(X[j], X[i-1]);
        swap(Y[j], Y[i-1]);
    }

    // Partition into mini-batches
    size_t num_batches = (m - 1) / batch_size + 1;
    std::vector<std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>>> mini_batches(num_batches);
    for(size_t i = 0; i < n; i++){
        mini_batches[i % num_batches].first.push_back(X[i]);
        mini_batches[i % num_batches].second.push_back(Y[i]);
    }

    return mini_batches;
}