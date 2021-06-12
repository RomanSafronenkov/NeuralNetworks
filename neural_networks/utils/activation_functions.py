import numpy as np
from typing import Union


def sigmoid(z: Union[np.array, float, int, list]) -> Union[np.array, float]:
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: Union[np.array, float, int, list]) -> Union[np.array, float]:
    """
    Sigmoid function derivative
    """
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)


def tanh(z: Union[np.array, float, int, list]) -> Union[np.array, float]:
    """
    Tanh function
    """
    return np.tanh(z)


def tanh_derivative(z: Union[np.array, float, int, list]) -> Union[np.array, float]:
    """
    Tanh function derivative
    """
    return 1 - np.tanh(z) ** 2


def relu(z: Union[np.array, float, int, list]) -> Union[np.array, float, int]:
    """
    ReLU function
    """
    return np.maximum(0, z)


def relu_derivative(z: Union[np.array, float, int, list]) -> np.array:
    """
    ReLU function derivative
    """
    return (z > 0).astype(int)


def softmax(z: np.array) -> np.array:
    """
    Softmax function
    """
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


def softmax_grad(z: np.array, y_true: np.array) -> np.array:
    """
    Softmax function derivative
    :param z: scores of the model
    :param y_true: true labels
    """
    s = softmax(z)
    grad = np.zeros(shape=s.shape)
    for obj, target in enumerate(y_true):
        for z in range(len(s[obj])):
            if target == z:
                grad[obj][z] = s[obj][z] * (1 - s[obj][z])
            else:
                grad[obj][z] = -s[obj][target] * s[obj][z]
    return grad
