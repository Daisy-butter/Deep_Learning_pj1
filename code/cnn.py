import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import pickle
import argparse
import time
import psutil  # For memory usage monitoring

# Utility Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-10)
    return np.sum(log_likelihood) / m

def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == y_true)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, filename):
    ensure_dir(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Data Loading
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    return images, labels

def preprocess_data(debug=False):
    if os.path.exists('dataset/mnist_preprocessed.npz'):
        try:
            data = np.load('dataset/mnist_preprocessed.npz')
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            X_test, y_test = data['X_test'], data['y_test']
            if debug:
                X_train, y_train = X_train[:5000], y_train[:5000]
                X_val, y_val = X_val[:1000], y_val[:1000]
                X_test, y_test = X_test[:1000], y_test[:1000]
            return X_train, y_train, X_val, y_val, X_test, y_test
        except Exception as e:
            print(f"Error loading preprocessed data: {e}. Reprocessing data...")

    ensure_dir('dataset/MNIST')
    try:
        X_train, y_train = load_mnist('dataset/MNIST', kind='train')
        X_test, y_test = load_mnist('dataset/MNIST', kind='t10k')
    except FileNotFoundError:
        raise FileNotFoundError("MNIST data files not found in 'dataset/MNIST'. "
                              "Download from http://yann.lecun.com/exdb/mnist/ and place in 'dataset/MNIST'.")

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    val_size = int(X_train.shape[0] * 0.2)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    if debug:
        X_train, y_train = X_train[:5000], y_train[:5000]
        X_val, y_val = X_val[:1000], y_val[:1000]
        X_test, y_test = X_test[:1000], y_test[:1000]

    try:
        np.savez('dataset/mnist_preprocessed.npz', 
                 X_train=X_train, y_train=y_train, 
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)
    except Exception as e:
        print(f"Warning: Failed to save preprocessed data: {e}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# CNN Layers
class Conv2D:
    def __init__(self, input_channels, num_filters, kernel_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(num_filters, input_channels, kernel_size, kernel_size) * np.sqrt(2. / (input_channels * kernel_size * kernel_size))
        self.biases = np.zeros((num_filters, 1))
        self.input = None
        self.output = None
    
    def forward(self, input):
        self.input = input
        batch_size, input_channels, input_height, input_width = input.shape
        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        if self.padding > 0:
            padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            padded_input = input
        
        output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                input_slice = padded_input[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.num_filters):
                    output[:, k, i, j] = np.sum(input_slice * self.weights[k, :, :, :], axis=(1, 2, 3)) + self.biases[k]
        
        self.output = output
        return output
    
    def backward(self, d_output, learning_rate):
        batch_size, _, output_height, output_width = d_output.shape
        _, input_channels, input_height, input_width = self.input.shape
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)
        
        if self.padding > 0:
            padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            padded_d_input = np.pad(d_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            padded_input = self.input
            padded_d_input = d_input
        
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                input_slice = padded_input[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.num_filters):
                    d_weights[k] += np.sum(input_slice * d_output[:, k, i, j].reshape(-1, 1, 1, 1), axis=0)
                    d_biases[k] += np.sum(d_output[:, k, i, j])
                    padded_d_input[:, :, h_start:h_end, w_start:w_end] += self.weights[k] * d_output[:, k, i, j].reshape(-1, 1, 1, 1)
        
        if self.padding > 0:
            d_input = padded_d_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = padded_d_input
        
        self.weights -= learning_rate * d_weights / batch_size
        self.biases -= learning_rate * d_biases / batch_size
        return d_input

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None
        self.max_indices = None
    
    def forward(self, input):
        self.input = input
        batch_size, channels, input_height, input_width = input.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channels, output_height, output_width))
        max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=np.int32)
        
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                input_slice = input[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(input_slice, axis=(2, 3))
                for b in range(batch_size):
                    for c in range(channels):
                        max_idx = np.unravel_index(np.argmax(input_slice[b, c]), (self.pool_size, self.pool_size))
                        max_indices[b, c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        self.output = output
        self.max_indices = max_indices
        return output
    
    def backward(self, d_output):
        batch_size, channels, output_height, output_width = d_output.shape
        d_input = np.zeros_like(self.input)
        for i in range(output_height):
            for j in range(output_width):
                for b in range(batch_size):
                    for c in range(channels):
                        h, w = self.max_indices[b, c, i, j]
                        d_input[b, c, h, w] += d_output[b, c, i, j]
        return d_input

class Flatten:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backward(self, d_output):
        return d_output.reshape(self.input_shape)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None
    
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output
    
    def backward(self, delta, learning_rate, l2_lambda=0.0):
        d_weights = np.dot(self.input.T, delta) + l2_lambda * self.weights
        d_biases = np.sum(delta, axis=0, keepdims=True)
        d_input = np.dot(delta, self.weights.T)
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        return d_input

# CNN Model
class CNN:
    def __init__(self, input_shape, num_classes):
        self.layers = []
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers.append(Conv2D(input_shape[0], 32, 3, padding=1))
        self.layers.append(MaxPool2D(2, 2))
        self.layers.append(Conv2D(32, 64, 3, padding=1))
        self.layers.append(MaxPool2D(2, 2))
        self.layers.append(Flatten())
        self.layers.append(Dense(64*7*7, 128))
        self.layers.append(Dense(128, num_classes))
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
    
    def forward(self, X):
        if len(X.shape) == 2:
            X = X.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return softmax(output)
    
    def backward(self, X, y, learning_rate, l2_lambda=0.0):
        output = self.forward(X)
        m = X.shape[0]
        delta = output
        delta[range(m), y] -= 1
        delta /= m
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                delta = layer.backward(delta, learning_rate, l2_lambda)
            else:
                delta = layer.backward(delta, learning_rate) if isinstance(layer, Conv2D) else layer.backward(delta)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.01, batch_size=32, l2_lambda=0.0, early_stopping=5, debug=False):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            start_time = time.time()
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i+batch_size].reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
                y_batch = y_train_shuffled[i:i+batch_size]
                self.backward(X_batch, y_batch, learning_rate, l2_lambda)
                if (i // batch_size + 1) % 100 == 0:
                    print(f'Epoch {epoch+1}, Batch {i // batch_size + 1}/{X_train.shape[0] // batch_size}, '
                          f'Memory Usage: {psutil.virtual_memory().percent}%')
            
            # Use subset for metrics if debug mode
            train_subset = X_train[:5000] if debug else X_train
            y_train_subset = y_train[:5000] if debug else y_train
            val_subset = X_val[:1000] if debug else X_val
            y_val_subset = y_val[:1000] if debug else y_val

            train_pred = self.forward(train_subset)
            train_loss = cross_entropy_loss(train_pred, y_train_subset)
            train_acc = accuracy(train_pred, y_train_subset)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            
            val_pred = self.forward(val_subset)
            val_loss = cross_entropy_loss(val_pred, y_val_subset)
            val_acc = accuracy(val_pred, y_val_subset)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {time.time() - start_time:.2f}s')
            
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == early_stopping:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def evaluate(self, X, y):
        pred = self.forward(X)
        loss = cross_entropy_loss(pred, y)
        acc = accuracy(pred, y)
        return loss, acc

# Visualization Functions
def visualize_weights(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            weights = layer.weights
            num_filters = weights.shape[0]
            plt.figure(figsize=(10, 5))
            for i in range(min(num_filters, 32)):
                plt.subplot(4, 8, i+1)
                plt.imshow(weights[i, 0], cmap='gray')
                plt.axis('off')
            plt.suptitle('First Conv Layer Filters')
            plt.savefig('conv_filters.png')
            plt.close()
            break

def visualize_predictions(model, X_test, y_test, num_samples=10):
    indices = np.random.choice(len(X_test), num_samples)
    samples = X_test[indices].reshape(-1, model.input_shape[0], model.input_shape[1], model.input_shape[2])
    labels = y_test[indices]
    preds = model.predict(samples)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(samples[i, 0], cmap='gray')
        plt.title(f'True: {labels[i]}\nPred: {preds[i]}')
        plt.axis('off')
    plt.savefig('predictions.png')
    plt.close()

def plot_training_history(model):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_history, label='Train Loss')
    plt.plot(model.val_loss_history, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(model.train_acc_history, label='Train Accuracy')
    plt.plot(model.val_acc_history, label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Main Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with smaller dataset')
    args = parser.parse_args()
    
    # Load and preprocess data
    start_time = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(debug=args.debug)
    print(f'Data preprocessing time: {time.time() - start_time:.2f}s')
    
    # Initialize and train CNN
    cnn = CNN(input_shape=(1, 28, 28), num_classes=10)
    cnn.train(X_train.reshape(-1, 1, 28, 28), y_train, 
              X_val.reshape(-1, 1, 28, 28), y_val,
              epochs=args.epochs, learning_rate=args.lr, 
              batch_size=args.batch_size, l2_lambda=0.0001, debug=args.debug)
    
    # Evaluate on test set
    test_loss, test_acc = cnn.evaluate(X_test.reshape(-1, 1, 28, 28), y_test)
    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Save model
    save_model(cnn, 'models/cnn_model.pkl')
    
    # Visualize results
    visualize_weights(cnn)
    visualize_predictions(cnn, X_test, y_test)
    plot_training_history(cnn)