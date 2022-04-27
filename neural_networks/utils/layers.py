import copy
from abc import ABC, abstractmethod

import numpy as np

from neural_networks.utils.activation_functions import softmax, softmax_grad


class BaseLayer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, x: np.array, grad: bool = True) -> np.array:
        pass

    @abstractmethod
    def backward(self, output_error: np.array) -> np.array:
        pass


class Linear(BaseLayer):
    """
    Linear class permorms ordinary FC layer in neural networks
    Parameters:
    n_input - size of input neurons
    n_output - size of output neurons
    Methods:
    forward(x) - performs forward pass of the layer
    backward(output_error, learning_rate) - performs backward pass of the layer
    """

    def __init__(self, n_input: int, n_output: int) -> None:
        self.input = None
        self.n_input = n_input
        self.n_output = n_output
        self.w = np.random.normal(scale=np.sqrt(2 / (n_input + n_output)), size=(n_input, n_output))
        self.b = np.random.normal(scale=np.sqrt(2 / (n_input + n_output)), size=(1, n_output))

        self.w_optimizer = None
        self.b_optimizer = None

    def set_optimizer(self, optimizer) -> None:
        self.w_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

        self.w_optimizer.set_weight(self.w)
        self.b_optimizer.set_weight(self.b)

    def forward(self, x: np.array, grad=True) -> np.array:
        self.input = x
        return x.dot(self.w) + self.b

    def backward(self, output_error: np.array) -> np.array:
        assert self.w_optimizer is not None and self.b_optimizer is not None, 'You should set an optimizer'
        w_grad = self.input.T.dot(output_error)
        b_grad = np.ones((1, len(output_error))).dot(output_error)
        input_error = output_error.dot(self.w.T)

        self.w = self.w_optimizer.step(w_grad)
        self.b = self.b_optimizer.step(b_grad)
        return input_error


class Activation(BaseLayer):
    """
    Activation class is used for activation function of the FC layer
    Params:
    activation_function - activation function (e.g. sigmoid, RElU, tanh)
    activation_derivative - derivative of the activation function
    Methods:
    forward(x) - performs forward pass of the layer
    backward(output_error, learning_rate) - performs backward pass of the layer
    """

    def __init__(self, activation_function: callable, activation_derivative: callable) -> None:
        self.input = None
        self.activation = activation_function
        self.derivative = activation_derivative

    def forward(self, x: np.array, grad=True) -> np.array:
        self.input = x
        return self.activation(x)

    def backward(self, output_error: np.array) -> np.array:
        return output_error * self.derivative(self.input)


class DropOut(BaseLayer):
    def __init__(self, p):
        self.input = None
        self.p = p
        self.q = 1 / (1 - p)
        self.mask = None

    def forward(self, x: np.array, grad=True) -> np.array:
        self.input = x
        if grad:
            self.mask = np.random.uniform(0, 1, size=x.shape) > self.p
            return self.input * self.q * self.mask

        return self.input

    def backward(self, output_error: np.array) -> np.array:
        return output_error * self.q * self.mask


class SoftMaxLayer(BaseLayer):
    """
    Class for softmax layer, it needs y_true for initialization, all the rest are the same as "Activation" class
    Deprecated
    """

    def __init__(self, activation_function: callable, activation_derivative: callable, y: np.array) -> None:
        self.input = None
        self.activation = softmax
        self.derivative = softmax_grad
        self.y_true = y

    def forward(self, x: np.array, grad=True) -> np.array:
        self.input = x
        return self.activation(x)

    def backward(self, output_error: np.array) -> np.array:
        return output_error * self.derivative(self.input, self.y_true)
