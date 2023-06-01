import gzip
import pickle
import random

import numpy as np

from data_loader import get_multipliers, load_data, vectorized_word


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.multipliers = np.reshape(get_multipliers(), (3, 1))

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range(0, n, mini_batch_size):
                batch = training_data[k : k + mini_batch_size]
                X = np.concatenate([x for x, _ in batch], axis=1)
                Y = np.concatenate([y for _, y in batch], axis=1)
                mini_batches.append((X, Y))

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, mini_batch_size)

            if test_data:
                result = self.evaluate(test_data)
                persentage = result * 100 / n_test
                print(f"Epoch {epoch}: {result} / {n_test} = {persentage:.2f} %")
            else:
                print(f"Epoch {epoch}")

    def update_mini_batch(self, mini_batch, eta, mini_batch_size):
        x, y = mini_batch
        nabla_biases, nabla_weights = self.backprop(x, y)
        self.biases = [
            b - nb * eta / mini_batch_size for b, nb in zip(self.biases, nabla_biases)
        ]
        self.weights = [
            w - nw * eta / mini_batch_size for w, nw in zip(self.weights, nabla_weights)
        ]

    def backprop(self, x, y):
        _, n = x.shape

        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        nabla_weights = [np.zeros(w.shape) for w in self.weights]

        a = x
        activations: list[np.ndarray] = [a]
        zs = []

        for b, w in zip(self.biases, self.weights):
            b_matrix = np.tile(b, (1, n))
            z = np.dot(w, a) + b_matrix
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        delta = self.cost_derivative(a, y) * sigmoid_prime(z)
        nabla_biases[-1] = np.sum(delta, axis=1).reshape(-1, 1)
        nabla_weights[-1] = delta.dot(activations[-2].transpose())

        for l in range(2, len(self.sizes)):
            delta = self.weights[-l + 1].transpose().dot(delta) * sigmoid_prime(zs[-l])
            nabla_biases[-l] = np.sum(delta, axis=1).reshape(-1, 1)
            nabla_weights[-l] = delta.dot(activations[-l - 1].transpose())

        return nabla_biases, nabla_weights

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x) * self.multipliers), y)
            for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def main():
    print("loading data...")
    training_data, test_data = load_data()
    print("learning...")
    net = Network([390, 30, 3])
    net.SGD(training_data, 5, 30, 3.0, test_data=test_data)

    print("saving network...")
    with gzip.open("net.pkl.gz", "wb") as f:
        pickle.dump(net, f)


def load_net() -> Network:
    with gzip.open("net.pkl.gz", "rb") as f:
        return pickle.load(f)


def predict_language(net: Network, word: str) -> str:
    x = vectorized_word(word.lower())
    result = net.feedforward(x) * net.multipliers
    index = np.argmax(result)
    confidence = result[index][0] / float(sum(result))
    language = {0: "norwegian", 1: "english", 2: "both english and norwegian"}[index]

    print("\nraw output:")
    print(net.feedforward(x))
    print("\ntransformed output:")
    print(result)
    print(f'\n"{word}" is {confidence * 100:.2f}% {language}\n')


if __name__ == "__main__":
    net = load_net()
    while True:
        word = input("type a word: ")
        predict_language(net, word)
