import numpy
import random
from keras.src.datasets import mnist
import json

# Load and preprocess data
(train_x, train_y), (testd_x, testd_y) = mnist.load_data()
train_x = train_x.reshape(60000, 784)/255.0
testd_x = testd_x.reshape(10000, 784)/255.0

print(train_x[0])

def one_hot_encode(labels, num_classes=10):
    return numpy.eye(num_classes)[labels]

train_y = one_hot_encode(train_y)
testd_y = one_hot_encode(testd_y)

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Compute the output of the network for input `a`."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, batch_size, lr, test_data=None):
        """Train the network using stochastic gradient descent."""
        print("Training started...")
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + batch_size] for k in range(0, n, batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, lr)

            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_batch(self, batch, lr):
        """Update weights and biases using backpropagation."""
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_b, delta_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_w)]

        self.weights = [w - (lr / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Perform backpropagation to compute gradients."""
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.outer(delta, activations[-2])

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = numpy.dot(self.weights[-l + 1].T, delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.outer(delta, activations[-l - 1])

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Evaluate the network's performance on test data."""
        results = [(numpy.argmax(self.feedforward(x)), numpy.argmax(y)) for x, y in test_data]
        return sum(int(pred == y) for pred, y in results)

# Helper functions
def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Initialize and train the network
net = Network([784, 30, 10])
net.SGD(list(zip(train_x, train_y)), 15, 10, 3.0, test_data=list(zip(testd_x, testd_y)))

