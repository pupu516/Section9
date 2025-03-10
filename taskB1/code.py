###---------------------------------
### Part a
# Task 1A: Warm Up
# This is a conceptual explanation and does not involve actual code execution.

# Forward Pass:
# Input Layer (X_i) -> Hidden Layer 1 -> Hidden Layer 2 -> Hidden Layer 3 -> Output Layer (y_hat)

# Backward Pass:
# Output Layer (y_hat) -> Hidden Layer 3 -> Hidden Layer 2 -> Hidden Layer 1 -> Input Layer (X_i)

# Dimensions for a 4-layer MLP:
# Input Layer: Dimension A (e.g., 784 for MNIST)
# Hidden Layer 1: Dimension B (e.g., 128)
# Hidden Layer 2: Dimension C (e.g., 64)
# Hidden Layer 3: Dimension D (e.g., 32)
# Output Layer: Dimension E (e.g., 10 for MNIST classification)

# Weights and Biases Dimensions:
# W1: (A, B), b1: (B,)
# W2: (B, C), b2: (C,)
# W3: (C, D), b3: (D,)
# W4: (D, E), b4: (E,)

# Example dimensions for MNIST:
# A = 784 (input dimension for MNIST images)
# B = 128 (hidden layer 1)
# C = 64 (hidden layer 2)
# D = 32 (hidden layer 3)
# E = 10 (output dimension for 10 classes)

# Forward Pass:
# X_i (A) -> W1 (A, B) + b1 (B) -> ReLU -> W2 (B, C) + b2 (C) -> ReLU -> W3 (C, D) + b3 (D) -> ReLU -> W4 (D, E) + b4 (E) -> y_hat (E)

# Backward Pass:
# Gradients are computed using the chain rule and propagated backward through the network.
# dW4, db4 -> dW3, db3 -> dW2, db2 -> dW1, db1





###---------------------------------
### Part b
import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.weights.append(np.random.randn(dims[i], dims[i + 1])
            self.biases.append(np.random.randn(dims[i + 1]))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, X, activation='relu'):
        self.activations = [X]
        self.z_values = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.activations[-1], W) + b
            self.z_values.append(z)
            if i == len(self.weights) - 1:
                # Output layer (no activation)
                self.activations.append(z)
            else:
                if activation == 'relu':
                    self.activations.append(self.relu(z))
                elif activation == 'sigmoid':
                    self.activations.append(self.sigmoid(z))
        return self.activations[-1]
    
    def backward(self, y, activation='relu'):
        m = y.shape[0]
        self.dW = []
        self.db = []
        delta = self.activations[-1] - y
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0) / m
            self.dW.insert(0, dW)
            self.db.insert(0, db)
            if i > 0:
                if activation == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i - 1])
                elif activation == 'sigmoid':
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i - 1])
    
    def update_parameters(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.dW[i]
            self.biases[i] -= learning_rate * self.db[i]


###---------------------------------
