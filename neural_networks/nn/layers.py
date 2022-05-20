import copy
from abc import ABC, abstractmethod
from typing import Tuple

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
    set_optimizer(optimizer) - is used for setting an optimizer for gradient descent
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

    def forward(self, x: np.array, grad: bool = True) -> np.array:
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

    def forward(self, x: np.array, grad: bool = True) -> np.array:
        self.input = x
        return self.activation(x)

    def backward(self, output_error: np.array) -> np.array:
        return output_error * self.derivative(self.input)


class DropOut(BaseLayer):
    """
    DropOut class is used for DropOut layer
    p – probability of an element to be zeroed. Default: 0.5

    https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """

    def __init__(self, p):
        super().__init__()
        self.input = None
        self.p = p
        self.q = 1 / (1 - p)
        self.mask = None

    def forward(self, x: np.array, grad: bool = True) -> np.array:
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

    https://arxiv.org/pdf/1502.03167.pdf'
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
            self.axis = 0
        elif num_dims == 4:
            shape = (1, num_features, 1, 1)
            self.axis = (0, 2, 3)
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

            mean = x.mean(axis=self.axis, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=self.axis, keepdims=True)

            self.x_centered = (x - mean)
            self.x_std = np.sqrt(var + self.eps)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var

        y = self.gamma * x_hat + self.beta
        return y

    def backward(self, output_error: np.array) -> np.array:
        assert self.gamma_optimizer is not None, 'You should set an optimizer'

        gamma_grad = np.sum(output_error * self.x_centered / self.x_std, axis=self.axis, keepdims=True)
        batch_size = output_error.shape[0]

        # following lines are got from the original paper, they are the same as final version for input error
        #
        # dldx = output_error * self.gamma
        # dldsigma2 = np.sum(dldx * self.x_centered * (-1/2) * (self.x_std ** -3), axis=0)
        # dldmu = np.sum((dldx * (-1/self.x_std)), axis=0) + dldsigma2 * np.sum(-2*self.x_centered, axis=0) / batch_size
        # input_error = dldx / self.x_std + dldsigma2 * 2 * self.x_centered / batch_size + dldmu / batch_size

        input_error = (1 / batch_size) * self.gamma / self.x_std * \
                      (batch_size * output_error - np.sum(output_error, axis=self.axis, keepdims=True) -
                       self.x_centered / self.x_std ** 2 * np.sum(output_error * self.x_centered, axis=self.axis,
                                                                  keepdims=True))

        self.gamma = self.gamma_optimizer.step(gamma_grad)
        if self.affine:
            beta_grad = np.sum(output_error, axis=self.axis, keepdims=True)
            self.beta = self.beta_optimizer.step(beta_grad)
        return input_error


class Conv2D(BaseLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0) -> None:
        super().__init__()
        self.in_channels = in_channels  # number of input channels
        self.out_channels = out_channels  # number of output channels
        self.kernel_size = kernel_size  # kernel size, int
        self.stride = stride  # stride, int
        self.padding = padding  # padding, int, only zeros padding

        limit = 1 / np.sqrt(self.kernel_size ** 2)
        self.kernel = np.random.uniform(
            -limit,
            limit,
            size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )

        self.bias = np.zeros((self.out_channels, 1))
        self.kernel_optimizer = None
        self.bias_optimizer = None

        self.input_tensor = None
        self.batch_size = None
        self.reformated_input = None
        self.reformated_kernel = None

        self.channel_steps = []

    def set_kernel(self, kernel: np.array) -> None:
        """
        For setting the kernel manually
        """
        assert len(kernel.shape) == 4
        self.kernel = kernel

    def set_optimizer(self, optimizer) -> None:
        self.kernel_optimizer = copy.copy(optimizer)
        self.bias_optimizer = copy.copy(optimizer)

        self.kernel_optimizer.set_weight(self.kernel)
        self.bias_optimizer.set_weight(self.bias)

    def _reformat_kernel(self) -> np.array:
        """
        Reformat the kernel to perform convolution as matrix multiplication
        """
        assert len(self.kernel.shape) == 4
        return self.kernel.reshape((self.out_channels, -1))

    def _reformat_input(self, input_tensor: np.array) -> np.array:
        """
        Reformat the batch of input images to perform convolution as matrix multiplication
        """
        result = []
        for image in input_tensor:
            result.append(self._reformat_image(image))

        return np.hstack(result)

    def _reformat_image(self, image: np.array) -> np.array:
        """
        Reformatting each image in the batch
        """
        result = []
        for channel in image:
            result.append(self._reformat_channel(channel))

        return np.vstack(result)

    def _reformat_channel(self, channel: np.array) -> np.array:
        """
        Reformat each channel in the image with addinction of zeros as padding
        """
        input_map = self._add_padding(channel)

        self.padded_width_in, self.padded_height_in = input_map.shape

        width_in, height_in = channel.shape

        self.width_out = (width_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.height_out = (height_in - self.kernel_size + 2 * self.padding) // self.stride + 1

        output_map = []

        write_steps = True
        if self.channel_steps:
            write_steps = False

        for i in range(self.width_out):
            for j in range(self.height_out):
                horizontal_slice_from = i * self.stride
                horizontal_slice_to = self.kernel_size + i * self.stride

                vertical_slice_from = j * self.stride
                vertical_slice_to = self.kernel_size + j * self.stride

                if write_steps:
                    self.channel_steps.append(
                        (horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to)
                    )

                subsample = input_map[horizontal_slice_from:horizontal_slice_to, vertical_slice_from:vertical_slice_to]
                output_map.append(subsample.reshape(-1, 1))

        return np.hstack(output_map)

    def _add_padding(self, input_tensor: np.array) -> np.array:
        new_width = input_tensor.shape[0] + self.padding * 2
        new_height = input_tensor.shape[1] + self.padding * 2
        padded_map = np.zeros((new_width, new_height))

        padded_map[self.padding:new_width - self.padding, self.padding:new_height - self.padding] = input_tensor
        return padded_map

    def _reformat_forward(self, forward_output: np.array) -> np.array:
        result = []
        for image in forward_output:
            result.append(image.reshape(self.batch_size, self.height_out, self.width_out))
        return np.hstack(result).reshape(self.batch_size, self.out_channels, self.height_out, self.width_out)

    @staticmethod
    def _reformat_output(output_error: np.array) -> np.array:
        images_result = []
        for image in output_error:
            channel_result = []
            for channel in image:
                channel_result.append(channel.ravel())
            images_result.append(np.vstack(channel_result))
        return np.hstack(images_result)

    def _format_image_back(self, image: np.array) -> np.array:
        zeros = np.zeros((self.in_channels, self.padded_width_in, self.padded_height_in))
        elements_in_kernel = self.kernel_size ** 2
        for i in range(self.in_channels):
            for j, line in enumerate(image.T):
                reshaped_conv = line[i * elements_in_kernel:(i + 1) * elements_in_kernel].reshape(self.kernel_size,
                                                                                                  self.kernel_size)

                horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to = self.channel_steps[
                    j]
                zeros[i][horizontal_slice_from:horizontal_slice_to,
                vertical_slice_from:vertical_slice_to] = reshaped_conv
        return zeros

    def _cut_padding(self, image: np.array) -> np.array:
        result = []
        for channel in image:
            channel_no_pad = channel[self.padding:self.padded_width_in - self.padding,
                             self.padding:self.padded_height_in - self.padding]
            result.append(channel_no_pad[np.newaxis, :])
        return np.vstack(result)

    def _reformat_input_error(self, input_error: np.array) -> np.array:
        size_for_one_image = int(len(input_error.T) / self.batch_size)
        result = []
        for i in range(self.batch_size):
            image = input_error[:, i * size_for_one_image:(i + 1) * size_for_one_image]
            image = self._format_image_back(image)

            if self.padding != 0:
                image = self._cut_padding(image)

            result.append(image[np.newaxis, :])
        return np.vstack(result)

    def forward(self, input_tensor: np.array, grad: bool = False) -> np.array:
        assert len(input_tensor.shape) == 4
        self.input_tensor = input_tensor
        self.batch_size = self.input_tensor.shape[0]

        self.reformated_input = self._reformat_input(input_tensor)
        self.reformated_kernel = self._reformat_kernel()

        result = self.reformated_kernel.dot(self.reformated_input) + self.bias
        return self._reformat_forward(result)

    def backward(self, output_error: np.array) -> np.array:
        assert self.kernel_optimizer is not None and self.bias_optimizer is not None, 'You should set an optimizer'
        output_error = self._reformat_output(output_error)
        w_grad = output_error.dot(self.reformated_input.T).reshape(self.kernel.shape)
        b_grad = output_error.dot(np.ones((output_error.shape[1], 1)))

        self.kernel_optimizer.step(w_grad)
        self.bias_optimizer.step(b_grad)

        input_error = self.reformated_kernel.T.dot(output_error)
        input_error = self._reformat_input_error(input_error)
        return input_error


class MaxPool2D(BaseLayer):
    def __init__(self, kernel_size: int, stride: int, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.input_tensor = None
        self.channel_steps = []

    def _reformat_input(self, input_tensor: np.array) -> np.array:
        result = []
        grad_result = []
        for image in input_tensor:
            forward_result, grad = self._reformat_image(image)
            result.append(forward_result)
            grad_result.append(grad)

        self.grad = grad_result
        return np.vstack(result)

    def _reformat_image(self, image: np.array) -> np.array:
        result = []
        grad_result = []
        for channel in image:
            forward_result, grad = self._reformat_channel(channel)
            result.append(forward_result)
            grad_result.append(grad)

        return np.vstack(result), grad_result

    def _reformat_channel(self, channel: np.array) -> Tuple[np.array, np.array]:
        input_map = self._add_padding(channel)

        width_in, height_in = channel.shape

        self.width_out = (width_in - self.kernel_size + 2 * self.padding) // self.stride + 1
        self.height_out = (height_in - self.kernel_size + 2 * self.padding) // self.stride + 1

        output_map = np.zeros((self.height_out, self.width_out))

        grad = []

        write_steps = True
        if self.channel_steps:
            write_steps = False

        for i in range(self.width_out):
            for j in range(self.height_out):
                horizontal_slice_from = i * self.stride
                horizontal_slice_to = self.kernel_size + i * self.stride

                vertical_slice_from = j * self.stride
                vertical_slice_to = self.kernel_size + j * self.stride

                if write_steps:
                    self.channel_steps.append(
                        (horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to)
                    )

                subsample = input_map[horizontal_slice_from:horizontal_slice_to, vertical_slice_from:vertical_slice_to]
                output_map[i][j] = np.max(subsample)

                grad.append(np.argmax(subsample))

        return output_map, grad

    def _add_padding(self, input_tensor: np.array) -> np.array:
        new_width = input_tensor.shape[0] + self.padding * 2
        new_height = input_tensor.shape[1] + self.padding * 2
        padded_map = np.zeros((new_width, new_height))

        padded_map[self.padding:new_width - self.padding, self.padding:new_height - self.padding] = input_tensor
        return padded_map

    def _rebuild_output_error(self, output_error: np.array) -> np.array:
        zeros = np.zeros(self.input_tensor.shape)
        for image_num, image in enumerate(output_error):
            for channel_num, channel in enumerate(image):
                for i in range(self.width_out * self.height_out):
                    horizontal_slice_from, horizontal_slice_to, vertical_slice_from, vertical_slice_to \
                        = self.channel_steps[i]

                    slice_ = zeros[image_num][channel_num][horizontal_slice_from:horizontal_slice_to,
                             vertical_slice_from:vertical_slice_to]
                    cur_max = self.grad[image_num][channel_num][i]

                    flatten_channel = channel.flatten()
                    flatten_zeros = slice_.flatten()

                    flatten_zeros[cur_max] = flatten_channel[i]

                    slice_ = flatten_zeros.reshape(self.kernel_size, self.kernel_size)
                    zeros[image_num][channel_num][horizontal_slice_from:horizontal_slice_to,
                    vertical_slice_from:vertical_slice_to] = slice_
        return zeros

    def forward(self, input_tensor: np.array, grad: bool = False) -> np.array:
        assert len(input_tensor.shape) == 4
        self.input_tensor = input_tensor

        result = self._reformat_input(input_tensor)
        return result.reshape(input_tensor.shape[0], input_tensor.shape[1], self.height_out, self.width_out)

    def backward(self, output_error: np.array) -> np.array:
        return self._rebuild_output_error(output_error)


class Ravel(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, input_tensor: np.array, grad: bool = False) -> np.array:
        length = len(input_tensor)
        self.input = input_tensor
        return input_tensor.reshape(length, -1)

    def backward(self, output_error: np.array) -> np.array:
        return output_error.reshape(self.input.shape)
