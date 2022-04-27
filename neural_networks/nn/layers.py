import copy
from abc import ABC, abstractmethod

import numpy as np


class BaseLayer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.array, grad: bool = True) -> np.array:
        return self.forward(x, grad)

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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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


class BatchNorm(BaseLayer):
    """
    num_features – number of features or channels CC of the input
    num_dims – number of input features
    eps – a value added to the denominator for numerical stability. Default: 1e-5
    momentum – the value used for the running_mean and running_var computation.
    Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
    affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True
    """

    def __init__(self, num_features, num_dims, eps=1e-05, momentum=0.1, affine=True):
        super().__init__()
        self.input = None
        self.x_centered = None
        self.x_std = None
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if num_dims == 2:
            shape = (1, num_features)
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
        else:
            raise ValueError("num_dims must be in (2, 4)")
        self.gamma = np.ones(shape=shape)
        self.beta = np.zeros(shape=shape)
        self.gamma_optimizer = None
        self.beta_optimizer = None

        self.moving_mean = np.zeros(shape=shape)
        self.moving_var = np.zeros(shape=shape)

    def set_optimizer(self, optimizer) -> None:
        self.gamma_optimizer = copy.copy(optimizer)
        self.gamma_optimizer.set_weight(self.gamma)
        if self.affine:
            self.beta_optimizer = copy.copy(optimizer)
            self.beta_optimizer.set_weight(self.beta)

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        if not grad:
            self.x_centered = (x - self.moving_mean)
            self.x_std = np.sqrt(self.moving_var + self.eps)
            x_hat = self.x_centered / self.x_std
        else:
            assert len(x.shape) in (2, 4)
            if len(x.shape) == 2:
                mean = x.mean(axis=0)
                var = ((x - mean) ** 2).mean(axis=0)
            else:
                raise NotImplementedError

            self.x_centered = (x - mean)
            self.x_std = np.sqrt(var + self.eps)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var

        y = self.gamma * x_hat + self.beta
        return y

    def backward(self, output_error: np.array) -> np.array:
        assert self.gamma_optimizer is not None, 'You should set an optimizer'

        gamma_grad = np.sum(output_error * self.x_centered / self.x_std, axis=0)
        batch_size = output_error.shape[0]

        input_error = (1 / batch_size) * self.gamma / self.x_std * \
                      (batch_size * output_error - np.sum(output_error, axis=0) -
                       self.x_centered / self.x_std ** 2 * np.sum(output_error * self.x_centered, axis=0))

        self.gamma = self.gamma_optimizer.step(gamma_grad)
        if self.affine:
            beta_grad = np.sum(output_error, axis=0)
            self.beta = self.beta_optimizer.step(beta_grad)
        return input_error


class Convolution2D(BaseLayer):
    """
    TODO
    """

    def __init__(self):
        pass

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        pass

    def backward(self, output_error: np.array) -> np.array:
        pass
