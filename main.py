import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('train.csv')
print(data)
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # Shuffle before splitting

# Split dataset into training and development sets
data_dev = data[:1000].T
print(data_dev)
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
_, m_train = X_train.shape  # Fix for missing variable

# Initialize parameters
def init_params():
    W1 = np.random.randn(10, 784) * 0.01  # Small random values
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

# Activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability trick
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivative of ReLU
def ReLU_deriv(Z):
    return Z > 0

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))  # 10 classes (0-9)
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]  # Number of samples
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

# Predictions and accuracy
def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

# Gradient Descent
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {acc:.4f}")

    return W1, b1, W2, b2

# Train model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)

# Visualization of activations
np.random.seed(0)
X_sample = np.random.rand(784, 1)  # Simulated input (28x28 image flattened)
W1_sample = np.random.randn(10, 784) * 0.01
b1_sample = np.zeros((10, 1))
W2_sample = np.random.randn(10, 10) * 0.01
b2_sample = np.zeros((10, 1))

# Forward propagation for visualization
Z1_sample = W1_sample.dot(X_sample) + b1_sample
A1_sample = ReLU(Z1_sample)
Z2_sample = W2_sample.dot(A1_sample) + b2_sample
A2_sample = softmax(Z2_sample)

# Plot activations of neurons in hidden layer
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(10), Z1_sample.flatten(), color='blue', alpha=0.6)
plt.title("Hidden Layer Activations (Before ReLU)")
plt.xlabel("Neuron")
plt.ylabel("Activation Value")
plt.subplot(1, 2, 2)
plt.bar(range(10), A1_sample.flatten(), color='red', alpha=0.6)
plt.title("Hidden Layer Activations (After ReLU)")
plt.xlabel("Neuron")
plt.ylabel("Activation Value")

plt.tight_layout()
plt.show()

