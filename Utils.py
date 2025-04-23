import numpy as np

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