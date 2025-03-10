from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode labels
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot_encode(y_train, 10)
y_test_onehot = one_hot_encode(y_test, 10)

# Initialize MLP and CNN
mlp = MLP(input_dim=784, hidden_dims=[128, 64, 32], output_dim=10)
cnn = CNN(input_shape=(28, 28, 1), num_filters=8, filter_size=3, hidden_dims=[64, 32], output_dim=10)

# Training loop
def train(model, X_train, y_train, X_test, y_test, epochs=10, learning_rate=0.01, activation='relu'):
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(X_train, activation)
        loss = np.mean((y_pred - y_train) ** 2)
        train_loss.append(loss)
        
        # Backward pass
        model.backward(y_train, activation)
        model.update_parameters(learning_rate)
        
        # Test loss
        y_test_pred = model.forward(X_test, activation)
        test_loss.append(np.mean((y_test_pred - y_test) ** 2))
    
    return train_loss, test_loss

# Train MLP
mlp_train_loss, mlp_test_loss = train(mlp, X_train, y_train_onehot, X_test, y_test_onehot, epochs=10, learning_rate=0.01, activation='relu')

# Train CNN
cnn_train_loss, cnn_test_loss = train(cnn, X_train.reshape(-1, 28, 28, 1), y_train_onehot, X_test.reshape(-1, 28, 28, 1), y_test_onehot, epochs=10, learning_rate=0.01, activation='relu')

# Plot convergence
plt.plot(mlp_train_loss, label="MLP Train Loss")
plt.plot(mlp_test_loss, label="MLP Test Loss")
plt.plot(cnn_train_loss, label="CNN Train Loss")
plt.plot(cnn_test_loss, label="CNN Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.savefig("convergence.png")

# Confusion matrix
def plot_confusion_matrix(model, X_test, y_test, activation='relu'):
    y_pred = model.forward(X_test, activation)
    y_pred_labels = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

# Plot confusion matrix for MLP
plot_confusion_matrix(mlp, X_test, y_test, activation='relu')

# Plot confusion matrix for CNN
plot_confusion_matrix(cnn, X_test.reshape(-1, 28, 28, 1), y_test, activation='relu')
