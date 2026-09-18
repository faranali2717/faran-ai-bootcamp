import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: turns off oneDNN optimization warnings

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# dataset loading
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

x_train = x_train_raw.reshape(x_train_raw.shape[0], 784) / 255.0
x_test = x_test_raw.reshape(x_test_raw.shape[0], 784) / 255.0

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train = one_hot(y_train_raw)
y_test = one_hot(y_test_raw)

# model parameters
w1 = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
b1 = np.zeros((1, 128))
w2 = np.random.randn(128, 10) * np.sqrt(2.0 / 128)
b2 = np.zeros((1, 10))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(a2, y):
    m = y.shape[0]
    return -1/m * np.sum(y * np.log(a2 + 1e-15))

def compute_accuracy(a2, y_labels):
    predictions = np.argmax(a2, axis=1)
    return np.mean(y_labels == predictions)

def forward_pass(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1 
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2 
    a2 = softmax(z2)
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache 

def backward_pass(x, y, cache, w2):
    m = x.shape[0]
    z1, a1, z2, a2 = cache['z1'], cache['a1'], cache['z2'], cache['a2']
    
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m 
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dw1, db1, dw2, db2 

# training hyper-parameters
epochs = 20
batch_size = 64
learning_rate = 0.1

num_batches = int(np.ceil(x_train.shape[0] / batch_size))
loss_history = []

print("starting training...\n" + "-"*40)

for epoch in range(epochs):
    permutation = np.random.permutation(x_train.shape[0])
    x_shuffled = x_train[permutation]
    y_shuffled = y_train[permutation]
    
    epoch_loss = 0.0

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, x_train.shape[0])
        x_batch = x_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        a2, cache = forward_pass(x_batch, w1, b1, w2, b2)
        
        batch_loss = compute_loss(a2, y_batch)
        epoch_loss += batch_loss * (end - start)

        dw1, db1, dw2, db2 = backward_pass(x_batch, y_batch, cache, w2)

        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1 
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2 

    average_loss = epoch_loss / x_train.shape[0]
    loss_history.append(average_loss)

    a2_test, _ = forward_pass(x_test, w1, b1, w2, b2)
    test_acc = compute_accuracy(a2_test, y_test_raw) * 100
    print(f"epoch {epoch+1:02d}/{epochs:02d} :: loss: {average_loss:.4f} :: test accuracy: {test_acc:.2f}%")

# final predictions
a2_final_train, _ = forward_pass(x_train, w1, b1, w2, b2)
final_train_acc = compute_accuracy(a2_final_train, y_train_raw) * 100

a2_final_test, _ = forward_pass(x_test, w1, b1, w2, b2)
final_test_acc = compute_accuracy(a2_final_test, y_test_raw) * 100

print(f"\nfinal training accuracy : {final_train_acc:.2f}%")
print(f"final testing accuracy  : {final_test_acc:.2f}%")

# plot results
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='b', linewidth=2)
plt.title("training loss curve (2-layer mlp numpy)")
plt.xlabel("epoch")
plt.ylabel("categorical cross-entropy loss")
plt.grid(True)
plt.tight_layout()
plt.show()