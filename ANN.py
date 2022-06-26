from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

def mse(output: np.ndarray, Y: np.ndarray):
    return np.mean(np.power(output - Y, 2))

def d_mse(output: np.ndarray, Y: np.ndarray):
    return (output - Y) * 2 / len(Y)

class ActivationsF:
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_prime(Z):
        exp = np.exp(Z)
        return exp * (1 - exp)

    def tanh(x):
        return np.tanh(x);

    def tanh_prime(x):
        return 1-np.tanh(x)**2;

    def relu(x):
        return np.maximum(0,x)

    def relu_prime(x):
        return 1 * (x > 0) 

    def softmax(Z):
        Z -= np.max(Z)
        exp = np.exp(Z)
        return exp / np.sum(exp)

    def softmax_prime(Z):
        soft = ActivationsF.softmax(Z)
        return soft * (1 - soft)

class Layer(ABC):
    @abstractmethod
    def feedforward(self, input: np.ndarray) -> np.ndarray:...
    @abstractmethod
    def backward(self, out_grad: np.ndarray, lr: float) -> np.ndarray:...

class Dense(Layer):
    def __init__(self, n_input, n_output) -> None:
        self.weight = np.random.uniform(-1.,1.,size=(n_input, n_output))/np.sqrt(n_input * n_output)
        self.bios = 0

    def feedforward(self, input: np.ndarray):
        # self.last_input = np.append(input, [[1]*input.shape[0]], axis=1)
        self.last_input = input
        self.last_output = input.dot(self.weight)
        return self.last_output

    def backward(self, out_grad: np.ndarray, lr: float):
        grad = out_grad.dot(self.weight.T)
        dw = self.last_input.T.dot(out_grad)

        self.weight -= lr * dw
        self.bios -= lr * out_grad
        return grad

class typeActivation(Enum):
    NONE = '',
    TANH = 'tanh',
    SIGMOID = 'sigmoid',
    RELU = 'relu',
    SOFTMAX = 'softmax'

class Activation(Layer):
    def __init__(self, type: typeActivation = typeActivation.NONE, f: callable = lambda x : x):
        if type == typeActivation.NONE:
            self.f = f
        self.f = eval('ActivationsF.' + str(type.name).lower())
        self.df = eval('ActivationsF.' + str(type.name).lower() + '_prime')
    
    def feedforward(self, input: np.ndarray):
        self.last_input = input
        self.last_output = self.f(self.last_input)
        return self.last_output

    def backward(self, out_grad: np.ndarray, lr = None):
        return self.df(self.last_input) * out_grad

class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate

    def feedforward(self, input: np.ndarray):
        self.mask = np.random.binomial(1, self.rate, size=input.shape) / self.rate
        return input * self.mask

    def backward(self, out_grad: np.ndarray, lr: float) -> np.ndarray:
        return out_grad * self.mask