import pickle

import numpy as np

from ANN import Layer

class Model:
    def __init__(self, layers: list[Layer], cost, cost_prime, verbose = True):
        self.layers = layers
        self.cost = cost
        self.cost_prime = cost_prime
        self.verbose = verbose

    def add(self, layer: Layer):
        self.layers.append(layer)

    def predict(self, x_input):
        output = x_input
        for layer in self.layers:
            output = layer.feedforward(output)
        return output

    def fit(self, x_input, y_output, epochs=5):
        self.outputs = []
        for i in range(epochs):
            errors = 0
            for x_sample, y_sample in zip(x_input, y_output):
                self.outputs = [x_sample.reshape((1, x_input.shape[1]))]
                y_sample = y_sample.reshape((1, 10))
                for layer in self.layers:
                    self.outputs.append(layer.feedforward(self.outputs[-1]))
                errors += self.cost(self.outputs[-1], y_sample)
                grad = self.cost_prime(self.outputs[-1], y_sample)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, 0.1)
            
            errors *= 2 / x_input.shape[0]
            if self.verbose:
                print(f"error: {errors}")

    def evaluate(self, X, Y) -> float:
        correct = 0
        n_features = X.shape[1]
        for x_sample, y_sample in zip(X, Y):
            x_sample = x_sample.reshape((1, n_features))
            y_sample = y_sample.reshape((1, 10))
            output = self.predict(x_sample)
            if np.argmax(output) == np.argmax(y_sample): correct += 1
        return correct/X.shape[0]

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            obj = pickle.load(f)

        return obj

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
