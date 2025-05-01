import numpy as np
from config import config
import json

# Model parameter handling
def store_model_parameters(model, filename):
    model_data = {key: value.tolist() for key, value in model.items()}
    with open(f'{filename}.json', 'w') as file:
        json.dump(model_data, file)

def retrieve_model_parameters(filename):
    with open(f'{filename}.json', 'r') as file:
        model_data = json.load(file)
    return {key: np.array(value) for key, value in model_data.items()}

# Activation functions and their derivatives
def leaky_relu_activation(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def derivative_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)

def selu_activation(x, alpha=1.6733, lambda_=1.0507):
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def derivative_selu(x, alpha=1.6733, lambda_=1.0507):
    return lambda_ * np.where(x > 0, 1, alpha * np.exp(x))

def gelu_activation(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def derivative_gelu(x):
    return 0.5 * (1 + np.erf(x / np.sqrt(2))) + (x / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

def softmax_activation(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Loss computation
def calculate_loss(true_y, predicted_y, model, regularization_factor):
    reg_loss = regularization_factor / 2 * sum(np.sum(weights ** 2) for weights in [model['W1'], model['W2'], model['W3']])
    log_loss = -np.mean(np.sum(true_y * np.log(predicted_y + 1e-12), axis=1))
    return log_loss + reg_loss

def calculate_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

# Batch Normalization forward pass
def batch_normalization_forward(x, gamma, beta, epsilon=1e-5):
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    x_normalized = (x - batch_mean) / np.sqrt(batch_var + epsilon)
    out = gamma * x_normalized + beta
    cache = (x, x_normalized, batch_mean, batch_var, gamma, beta, epsilon)
    return out, cache

# Batch Normalization backward pass
def batch_normalization_backward(dout, cache):
    x, x_normalized, batch_mean, batch_var, gamma, beta, epsilon = cache
    m = x.shape[0]

    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_normalized = dout * gamma
    dvar = np.sum(dx_normalized * (x - batch_mean) * -0.5 * (batch_var + epsilon)**(-1.5), axis=0)
    dmean = np.sum(dx_normalized * -1 / np.sqrt(batch_var + epsilon), axis=0) + dvar * np.sum(-2 * (x - batch_mean), axis=0) / m
    dx = dx_normalized / np.sqrt(batch_var + epsilon) + dvar * 2 * (x - batch_mean) / m + dmean / m

    return dx, dgamma, dbeta

# Dropout implementation
def dropout(x, rate, training=True):
    if training:
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        return x * mask
    return x

# Setup the model with Batch Normalization and Dropout parameters
def setup_model(input_size, hidden_size, output_size):
    model = {
        'W1': np.random.randn(input_size, hidden_size) * 0.01,
        'b1': np.zeros(hidden_size),
        'gamma1': np.ones(hidden_size),
        'beta1': np.zeros(hidden_size),
        'W2': np.random.randn(hidden_size, hidden_size) * 0.01,
        'b2': np.zeros(hidden_size),
        'gamma2': np.ones(hidden_size),
        'beta2': np.zeros(hidden_size),
        'W3': np.random.randn(hidden_size, output_size) * 0.01,
        'b3': np.zeros(output_size),
    }
    return model

# Forward propagation with Batch Normalization and Dropout
def model_forward_with_bn(model, X, activation=config.activation, dropout_rate=0.5, training=True):
    activations = {}
    caches = {}

    # Retrieve parameters
    W1, b1, gamma1, beta1 = model['W1'], model['b1'], model['gamma1'], model['beta1']
    W2, b2, gamma2, beta2 = model['W2'], model['b2'], model['gamma2'], model['beta2']
    W3, b3 = model['W3'], model['b3']

    # Layer 1
    z1 = X @ W1 + b1
    a1_bn, bn_cache1 = batch_normalization_forward(z1, gamma1, beta1)
    if activation == 'leaky_relu':
        a1 = leaky_relu_activation(a1_bn)
    if activation == 'sigmoid':
        a1 = sigmoid_activation(a1_bn)
    if activation =='selu':
        a1 = selu_activation(a1_bn)
    if activation == 'gelu':
        a1 = gelu_activation(a1_bn)
    a1 = dropout(a1, dropout_rate, training)  # Apply dropout
    activations['a1'], caches['bn1'] = a1, bn_cache1

    # Layer 2
    z2 = a1 @ W2 + b2
    a2_bn, bn_cache2 = batch_normalization_forward(z2, gamma2, beta2)
    if activation == 'leaky_relu':
        a2 = leaky_relu_activation(a2_bn)
    if activation == 'sigmoid':
        a2 = sigmoid_activation(a2_bn)
    if activation =='selu':
        a2 = selu_activation(a2_bn)
    if activation == 'gelu':
        a2 = gelu_activation(a2_bn)
    a2 = dropout(a2, dropout_rate, training)  # Apply dropout
    activations['a2'], caches['bn2'] = a2, bn_cache2

    # Layer 3 (output layer)
    z3 = a2 @ W3 + b3
    a3 = softmax_activation(z3)
    activations['a3'] = a3

    return a3, activations, caches

def model_backward_with_bn(model, activations, caches, X, y, y_hat, reg_strength, activation=config.activation):
    gradients = {}
    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    gamma1, gamma2 = model['gamma1'], model['gamma2']

    # Backpropagation for Layer 3
    error = y_hat - y
    gradients['dW3'] = activations['a2'].T @ error + reg_strength * W3
    gradients['db3'] = np.sum(error, axis=0)

    # Backpropagation for Layer 2
    da2 = error @ W3.T
    if activation == 'leaky_relu':
        da2 *= derivative_leaky_relu(activations['a2'])
    if activation == 'sigmoid':
        da2 *= derivative_sigmoid(activations['a2'])
    if activation == 'selu':
        da2 *= derivative_selu(activations['a2'])
    if activation == 'gelu':
        da2 *= derivative_gelu(activations['a2'])
    dz2, dgamma2, dbeta2 = batch_normalization_backward(da2, caches['bn2'])
    gradients['dW2'] = activations['a1'].T @ dz2 + reg_strength * W2
    gradients['db2'] = np.sum(dz2, axis=0)
    gradients['dgamma2'], gradients['dbeta2'] = dgamma2, dbeta2

    # Backpropagation for Layer 1
    da1 = dz2 @ W2.T
    if activation == 'leaky_relu':
        da1 *= derivative_leaky_relu(activations['a1'])
    if activation == 'sigmoid':
        da1 *= derivative_sigmoid(activations['a1'])
    if activation == 'selu':
        da1 *= derivative_selu(activations['a1'])
    if activation == 'gelu':
        da1 *= derivative_gelu(activations['a1'])
    dz1, dgamma1, dbeta1 = batch_normalization_backward(da1, caches['bn1'])
    gradients['dW1'] = X.T @ dz1 + reg_strength * W1
    gradients['db1'] = np.sum(dz1, axis=0)
    gradients['dgamma1'], gradients['dbeta1'] = dgamma1, dbeta1

    return gradients

def update_parameters(model, gradients, learn_rate):
    for key in model:
        model[key] -= learn_rate * gradients.get(f'd{key}', 0)
    return model