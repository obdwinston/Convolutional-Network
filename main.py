import numpy as np
from os.path import join

from utils import split_data
from model import lenet_predict, lenet_train

test_ratio = 0.01
epochs = 5
alpha = 0.001 # learning rate
batch_size = 64

# load data
folder = './data'
X = np.load(join(folder, 'X_mnist.npy'))
y = np.load(join(folder, 'y_mnist.npy'))
X_train, y_train, X_test, y_test = split_data(X, y, test_ratio=test_ratio)

# train model
lenet_train(X_train, y_train, epochs=epochs, alpha=alpha, batch_size=batch_size)

# predict model
params = np.load('./params.npz')
lenet_predict(X_test, y_test, params)
