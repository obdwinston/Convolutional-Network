import numpy as np

def split_data(X, y, test_ratio=0.1):

    test_size = int(X.shape[0] * test_ratio)
    train_size = X.shape[0] - test_size
    indices = np.random.permutation(X.shape[0])
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    X_train, X_test = X[train_indices, :], X[test_indices, :]
    y_train, y_test = y[:, train_indices], y[:, test_indices]

    print('Training:', X_train.shape, y_train.shape)
    print('Testing:', X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test
