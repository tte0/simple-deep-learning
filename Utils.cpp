#include <vector>
#include <random>
using namespace std;

std::random_device rd;
std::mt19937 rng(rd());

vector<pair<vector<vector<double>>,vector<vector<double>>>>
random_mini_batches(vector<vector<double>> X, vector<vector<double>> Y, int batch_size){
    int n = X.size();
    int m = X[0].size();

    // Shuffle (X, Y)
    for(int i = n; i > 0; i--){
        int j = rng() % i;
        swap(X[j], X[i-1]);
        swap(Y[j], Y[i-1]);
    }

    // Partition into mini-batches
    int num_batches = (m - 1) / batch_size + 1;
    vector<pair<vector<vector<double>>,vector<vector<double>>>> mini_batches(num_batches);
    for(int i = 0; i < n; i++){
        mini_batches[i % num_batches].first.push_back(X[i]);
        mini_batches[i % num_batches].second.push_back(Y[i]);
    }

    return mini_batches;
}