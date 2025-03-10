class CNN:
    def __init__(self, input_shape, num_filters, filter_size, hidden_dims, output_dim):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Initialize convolutional layer weights and biases
        self.conv_weights = np.random.randn(num_filters, input_shape[2], filter_size, filter_size)
        self.conv_biases = np.random.randn(num_filters)
        
        # Initialize fully connected MLP
        self.mlp = MLP(input_shape[0] * input_shape[1] * num_filters, hidden_dims, output_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, X, activation='relu'):
        # Convolutional layer
        self.conv_output = np.zeros((X.shape[0], X.shape[1] - self.filter_size + 1, X.shape[2] - self.filter_size + 1, self.num_filters))
        for i in range(self.num_filters):
            for j in range(X.shape[1] - self.filter_size + 1):
                for k in range(X.shape[2] - self.filter_size + 1):
                    self.conv_output[:, j, k, i] = np.sum(X[:, j:j+self.filter_size, k:k+self.filter_size, :] * self.conv_weights[i], axis=(1, 2, 3)) + self.conv_biases[i]
        
        # ReLU activation
        if activation == 'relu':
            self.conv_output = self.relu(self.conv_output)
        elif activation == 'sigmoid':
            self.conv_output = self.sigmoid(self.conv_output)
        
        # Flatten and pass to MLP
        self.flattened = self.conv_output.reshape(X.shape[0], -1)
        return self.mlp.forward(self.flattened, activation)
    
    def backward(self, y, activation='relu'):
        # Backprop through MLP
        self.mlp.backward(y, activation)
        
        # Backprop through convolutional layer
        delta = self.mlp.dW[0].reshape(self.conv_output.shape)
        if activation == 'relu':
            delta *= self.relu_derivative(self.conv_output)
        elif activation == 'sigmoid':
            delta *= self.sigmoid_derivative(self.conv_output)
        
        # Update convolutional weights and biases
        self.dW_conv = np.zeros_like(self.conv_weights)
        self.db_conv = np.zeros_like(self.conv_biases)
        for i in range(self.num_filters):
            for j in range(self.filter_size):
                for k in range(self.filter_size):
                    self.dW_conv[i, j, k] = np.sum(delta[:, :, :, i] * self.conv_output[:, j:j+self.filter_size, k:k+self.filter_size, :], axis=(0, 1, 2))
            self.db_conv[i] = np.sum(delta[:, :, :, i])
    
    def update_parameters(self, learning_rate):
        self.mlp.update_parameters(learning_rate)
        self.conv_weights -= learning_rate * self.dW_conv
        self.conv_biases -= learning_rate * self.db_conv
