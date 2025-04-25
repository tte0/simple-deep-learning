#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>

std::vector<std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>>>
random_mini_batches(std::vector<std::vector<double>> X, std::vector<std::vector<double>> Y, size_t batch_size);

#endif // UTILS_H