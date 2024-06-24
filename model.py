import numpy as np
import matplotlib.pyplot as plt

def add_padding(A, p):

    return np.pad(A, ((0, 0), (p, p), (p, p), (0, 0)), 'constant', constant_values=0)

def forward_convolution(A_prev, W, b, p, s):
    '''
    A_prev (m, n_H_prev, n_W_prev, n_C_prev)
    W (f, f, n_C_prev, n_C)
    b (1, 1, 1, n_C)
    '''

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    n_H = int((n_H_prev + 2 * p - f) / s) + 1
    n_W = int((n_W_prev + 2 * p - f) / s) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = add_padding(A_prev, p)

    # convolve
    for i in range(m):
        a_prev_pad = A_prev_pad[i]

        for h in range(n_H):
            h_start = h * s
            h_end = h_start + f
            
            for w in range(n_W):
                w_start = w * s
                w_end = w_start + f
                
                for c in range(n_C):
                    a_prev_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]
                    
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = np.sum(np.multiply(a_prev_slice, weights)) + biases
    
    # activate @NOTE
    A = np.tanh(Z) # tanh activation
    # A = 1 / (1 + np.exp(-Z)) # sigmoid activation
    # A = np.maximum(0, Z) # relu activation

    cache = (A_prev, W, b, p, s, Z)

    return A, cache

def backward_convolution(A, dA, cache):

    # derivative @NOTE
    dAdZ = 1 - A ** 2 # tanh derivative
    # dAdZ = A * (1 - A) # sigmoid derivative
    # dAdZ = np.where(A > 0, 1, 0) # relu derivative
    dZ = dA * dAdZ

    (A_prev, W, b, p, s, Z) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = add_padding(A_prev, p)
    dA_prev_pad = add_padding(dA_prev, p)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            h_start = h * s
            h_end = h_start + f
            
            for w in range(n_W):
                w_start = w * s
                w_end = w_start + f
                
                for c in range(n_C):
                    a_prev_pad_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]

                    da_prev_pad[h_start:h_end, w_start:w_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_prev_pad_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[p:-p, p:-p, :] if p != 0 else da_prev_pad

    return A_prev, dA_prev, dW, db

def forward_pooling(A_prev, f, s):

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    n_H = int((n_H_prev - f) / s) + 1
    n_W = int((n_W_prev - f) / s) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev = A_prev[i]

        for h in range(n_H):
            h_start = h * s
            h_end = h_start + f
            
            for w in range(n_W):
                w_start = w * s
                w_end = w_start + f
                
                for c in range(n_C):
                    a_prev_slice = a_prev[h_start:h_end, w_start:w_end, c]
                    
                    A[i, h, w, c] = np.mean(a_prev_slice) # average pooling
                    # A[i, h, w, c] = np.max(a_prev_slice) # maximum pooling @NOTE

    cache = (A_prev, f, s)
    
    return A, cache

def backward_pooling(A, dA, cache):

    (A_prev, f, s) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

    for i in range(m):

        for h in range(n_H):
            h_start = h * s
            h_end = h_start + f
            
            for w in range(n_W):
                w_start = w * s
                w_end = w_start + f

                for c in range(n_C):

                    # average pooling
                    da = dA[i, h, w, c]

                    shape = (f, f)
                    average = da / (f * f)
                    da_prev = np.full(shape, average)

                    dA_prev[i, h_start:h_end, w_start:w_end, c] += da_prev

                    # maximum pooling @NOTE
                    # da = dA[i, h, w, c]

                    # a_prev = A_prev[i]
                    # a_prev_slice = a_prev[h_start:h_end, w_start:w_end, c]
                    # da_prev = (a_prev_slice == np.max(a_prev_slice)) * da
                    
                    # dA_prev[i, h_start:h_end, w_start:w_end, c] += da_prev
    
    return A_prev, dA_prev

def forward_layer(A_prev, W, b):

    Z = np.dot(W, A_prev) + b

    # activate @NOTE
    A = np.tanh(Z) # tanh activation
    # A = 1 / (1 + np.exp(-Z)) # sigmoid activation
    # A = np.maximum(0, Z) # relu activation

    cache = (A_prev, W, b, Z)

    return A, cache

def backward_layer(A, dA, cache):

    (A_prev, W, b, Z) = cache

    m = A_prev.shape[0]

    # derivative @NOTE
    dAdZ = 1 - A ** 2 # tanh derivative
    # dAdZ = A * (1 - A) # sigmoid derivative
    # dAdZ = np.where(A > 0, 1, 0) # relu derivative
    dZ = dA * dAdZ
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    return A_prev, dA_prev, dW, db

def create_batches(X, y, batch_size):

    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffled = X[permutation, :]
    y_shuffled = y[:, permutation]

    mini_batches = []
    n_batches = m // batch_size
    assert n_batches != 0, 'Number of batches is 0'

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        X_batch = X_shuffled[start:end, :]
        y_batch = y_shuffled[:, start:end]

        mini_batches.append((X_batch, y_batch))
    
    return mini_batches

def lenet_train(X, y, epochs=10, alpha=0.05, batch_size=64):

    epsilon = 1e-8
    costs = []
    accuracies = []

    (m, n_H_prev, n_W_prev, n_C_prev) = X.shape

    W_c1 = np.random.randn(5, 5, 1, 6) * np.sqrt(2 / (5 * 5 * 1))
    b_c1 = np.zeros((1, 1, 1, 6))

    W_c2 = np.random.randn(5, 5, 6, 16) * np.sqrt(2 / (5 * 5 * 6))
    b_c2 = np.zeros((1, 1, 1, 16))

    W_fc1 = np.random.randn(120, 5 * 5 * 16) * np.sqrt(2 / (5 * 5 * 16))
    b_fc1 = np.zeros((120, 1))

    W_fc2 = np.random.randn(84, 120) * np.sqrt(2 / 120)
    b_fc2 = np.zeros((84, 1))

    W_fc3 = np.random.randn(10, 84) * np.sqrt(2 / 84)
    b_fc3 = np.zeros((10, 1))

    # params = np.load('./params.npz')

    # costs = list(params['costs'])
    # accuracies = list(params['accuracies'])
    # W_c1 = params['W_c1']
    # W_c2 = params['W_c2']
    # W_fc1 = params['W_fc1']
    # W_fc2 = params['W_fc2']
    # W_fc3 = params['W_fc3']
    # b_c1 = params['b_c1']
    # b_c2 = params['b_c2']
    # b_fc1 = params['b_fc1']
    # b_fc2 = params['b_fc2']
    # b_fc3 = params['b_fc3']

    for i in range(epochs):

        mini_batches = create_batches(X, y, batch_size)
        batches = len(mini_batches)

        for j, mini_batch in enumerate(mini_batches):

            (X_batch, y_batch) = mini_batch
            m = X_batch.shape[0]

            # forward propagation

            # hidden layers
            A, cache_c1 = forward_convolution(X_batch, W_c1, b_c1, p=2, s=1)
            A, cache_p1 = forward_pooling(A, f=2, s=2)
            A, cache_c2 = forward_convolution(A, W_c2, b_c2, p=0, s=1)
            A, cache_p2 = forward_pooling(A, f=2, s=2)

            A = A.reshape(m, 5 * 5 * 16) # flatten
            A = A.T # transpose

            A, cache_fc1 = forward_layer(A, W_fc1, b_fc1)
            A, cache_fc2 = forward_layer(A, W_fc2, b_fc2)

            # softmax layer
            A, cache_fc3 = forward_layer(A, W_fc3, b_fc3)
            E = np.exp(A - np.max(A, axis=0, keepdims=True))
            A = E / np.sum(E, axis=0, keepdims=True)

            # cost computation
            cost = -(1 / m) * np.sum(y_batch * np.log(A + epsilon))
            y_pred = np.argmax(A, axis=0, keepdims=True)
            y_true = np.argmax(y_batch, axis=0, keepdims=True)
            accuracy = np.mean(y_pred == y_true)

            print(f'=== Epoch {i + 1} / {epochs}, Batch {j + 1} / {batches} ===')
            print(f'Cost: {cost: .10f}, Accuracy: {accuracy: .10f}')
            costs.append(cost)
            accuracies.append(accuracy)

            # backward propagation

            # softmax layer
            (A_prev, W, b, Z) = cache_fc3
            dZ = A - y_batch
            dW_fc3 = (1 / m) * np.dot(dZ, A_prev.T)
            db_fc3 = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            A = A_prev
            dA = np.dot(W.T, dZ)

            # hidden layers
            A, dA, dW_fc2, db_fc2 = backward_layer(A, dA, cache_fc2)
            A, dA, dW_fc1, db_fc1 = backward_layer(A, dA, cache_fc1)

            A = A.T # transpose
            A = A.reshape((m, 5, 5, 16)) # unflatten
            dA = dA.T # transpose
            dA = dA.reshape((m, 5, 5, 16)) # unflatten

            A, dA = backward_pooling(A, dA, cache_p2)
            A, dA, dW_c2, db_c2 = backward_convolution(A, dA, cache_c2)
            A, dA = backward_pooling(A, dA, cache_p1)
            A, dA, dW_c1, db_c1 = backward_convolution(A, dA, cache_c1)

            # update parameters

            W_c1 -= alpha * dW_c1
            W_c2 -= alpha * dW_c2
            W_fc1 -= alpha * dW_fc1
            W_fc2 -= alpha * dW_fc2
            W_fc3 -= alpha * dW_fc3

            b_c1 -= alpha * db_c1
            b_c2 -= alpha * db_c2
            b_fc1 -= alpha * db_fc1
            b_fc2 -= alpha * db_fc2
            b_fc3 -= alpha * db_fc3

        # save every epoch
        np.savez('params.npz', costs=np.array(costs), accuracies=np.array(accuracies),
                 W_c1=W_c1, W_c2=W_c2, W_fc1=W_fc1, W_fc2=W_fc2, W_fc3=W_fc3,
                 b_c1=b_c1, b_c2=b_c2, b_fc1=b_fc1, b_fc2=b_fc2, b_fc3=b_fc3)

def lenet_predict(X, y, params):

    (m, n_H_prev, n_W_prev, n_C_prev) = X.shape
        
    costs = params['costs']
    accuracies = params['accuracies']
    W_c1 = params['W_c1']
    W_c2 = params['W_c2']
    W_fc1 = params['W_fc1']
    W_fc2 = params['W_fc2']
    W_fc3 = params['W_fc3']
    b_c1 = params['b_c1']
    b_c2 = params['b_c2']
    b_fc1 = params['b_fc1']
    b_fc2 = params['b_fc2']
    b_fc3 = params['b_fc3']

    # forward propagation

    # hidden layers
    A, _ = forward_convolution(X, W_c1, b_c1, p=2, s=1)
    A, _ = forward_pooling(A, f=2, s=2)
    A, _ = forward_convolution(A, W_c2, b_c2, p=0, s=1)
    A, _ = forward_pooling(A, f=2, s=2)

    A = A.reshape(m, 5 * 5 * 16) # flatten
    A = A.T # transpose

    A, _ = forward_layer(A, W_fc1, b_fc1)
    A, _ = forward_layer(A, W_fc2, b_fc2)

    # softmax layer
    A, _ = forward_layer(A, W_fc3, b_fc3)
    E = np.exp(A - np.max(A, axis=0, keepdims=True))
    A = E / np.sum(E, axis=0, keepdims=True)

    y_pred = np.argmax(A, axis=0, keepdims=True)
    y_true = np.argmax(y, axis=0, keepdims=True)
    accuracy = np.mean(y_pred == y_true)
    print(f'Accuracy: {accuracy: .10f}')
    
    # plot results
    
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 5)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost', color='tab:blue')
    ax1.plot(costs, color='tab:blue', label='Cost')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Cost and Training Accuracy', fontweight='bold')
    ax1.grid('on')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Training Accuracy', color='tab:orange')
    ax2.plot(accuracies, color='tab:orange', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    random_indices = np.random.choice(m, 5, replace=False)
    for i, idx in enumerate(random_indices):
        ax = fig.add_subplot(gs[1, i])
        img = X[idx].reshape(28, 28)
        pred_label = y_pred[0, idx]
        true_label = y_true[0, idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {pred_label}, True: {true_label}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
