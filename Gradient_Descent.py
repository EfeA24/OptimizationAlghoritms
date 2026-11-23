import numpy as np

np.random.seed(42)

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([[2.0], [4.0], [6.0], [8.0]])

W1 = np.random.randn(1, 3) * 0.1
b1 = np.zeros((1, 3))

W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

learning_rate = 0.01
epochs = 3000
n = X.shape[0]


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


for epoch in range(epochs):
    h1 = X @ W1 + b1
    z1 = relu(h1)
    h2 = z1 @ W2 + b2

    loss = np.mean((h2 - y) ** 2)

    e2 = (2 / n) * (h2 - y)

    dW2 = z1.T @ e2
    db2 = np.sum(e2, axis=0, keepdims=True)

    g_z1 = e2 @ W2.T
    D1 = relu_derivative(h1)
    e1 = g_z1 * D1

    dW1 = X.T @ e1
    db1 = np.sum(e1, axis=0, keepdims=True)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")


h1 = X @ W1 + b1
z1 = relu(h1)
predictions = z1 @ W2 + b2

print("\nPredictions:")
print(predictions)

print("\nTargets:")
print(y)
